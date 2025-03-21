# coding:utf-8
# Modified by Yan Wang based on the following repositories.
# CRM_RGBTSeg: https://github.com/UkcheolShin/CRM_RGBTSeg

import os
import copy
import itertools
import pickle

import torch
import torch.nn as nn 
from torch.nn import functional as F
from pytorch_lightning import LightningModule
from sklearn.metrics import confusion_matrix
import torchvision.utils as vutils

from torchmetrics.classification import MulticlassPrecision, MulticlassRecall, MulticlassJaccardIndex

from util.util import compute_results, get_palette_MF, get_palette_PST, get_palette_FMB ,visualize_pred
from .registry import MODELS
from models.mask2former import RGBTMaskFormer
from detectron2.projects.deeplab import build_lr_scheduler
from detectron2.solver.build import maybe_add_gradient_clipping
import cv2
import numpy as np

@MODELS.register_module(name='RGBTMaskFormer')
class Model_RGBT_Mask2Former(LightningModule):
    def __init__(self, cfg):
        super(Model_RGBT_Mask2Former, self).__init__()
        self.save_hyperparameters()

        self.cfg = cfg
        self.num_classes = cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES
        self.learning_rate = cfg.SOLVER.BASE_LR
        self.lr_decay = cfg.SOLVER.WEIGHT_DECAY

        if self.num_classes == 9 : 
            self.label_list = ["unlabeled", "car", "person", "bike", "curve", "car_stop", "guardrail", "color_cone", "bump"]
            self.palette = get_palette_MF()
            self.val_precision = MulticlassPrecision(num_classes=self.num_classes,average=None, dist_sync_on_step=True)
            self.val_recall = MulticlassRecall(num_classes=self.num_classes,average=None, dist_sync_on_step=True)
            self.val_iou = MulticlassJaccardIndex(num_classes=self.num_classes,average=None, dist_sync_on_step=True)
        elif self.num_classes == 5 : 
            self.label_list = ["unlabeled", "fire_extinhuisher", "backpack", "hand_drill", "rescue_randy"]
            self.palette = get_palette_PST()
            self.val_precision = MulticlassPrecision(num_classes=self.num_classes,average=None, dist_sync_on_step=True)
            self.val_recall = MulticlassRecall(num_classes=self.num_classes,average=None, dist_sync_on_step=True)
            self.val_iou = MulticlassJaccardIndex(num_classes=self.num_classes,average=None, dist_sync_on_step=True)
        elif self.num_classes == 15 : 
            self.label_list = ["unlabeled","Road", "Sidewalk", "Building", "Traffic Lamp", "Traffic Sign", "Vegetation", 
                    "Sky", "Person", "Car", "Truck", "Bus", "Motorcycle", "Bicycle", "Pole"] 
            self.palette = get_palette_FMB()      
            self.val_precision = MulticlassPrecision(num_classes=self.num_classes,average=None, dist_sync_on_step=True, ignore_index= 0)
            self.val_recall = MulticlassRecall(num_classes=self.num_classes, average=None,dist_sync_on_step=True, ignore_index=0)
            self.val_iou = MulticlassJaccardIndex(num_classes=self.num_classes,average=None, dist_sync_on_step=True, ignore_index=0)

        self.network = RGBTMaskFormer(cfg)
        self.optimizer = self.build_optimizer(cfg, self.network)
        self.scheduler = self.build_lr_scheduler(cfg, self.optimizer)
        self.automatic_optimization = False

    @classmethod
    def build_lr_scheduler(cls, cfg, optimizer):
        """
        It now calls :func:`detectron2.solver.build_lr_scheduler`.
        Overwrite it if you'd like a different scheduler.
        """
        return build_lr_scheduler(cfg, optimizer)

    @classmethod
    def build_optimizer(cls, cfg, model):
        weight_decay_norm = cfg.SOLVER.WEIGHT_DECAY_NORM
        weight_decay_embed = cfg.SOLVER.WEIGHT_DECAY_EMBED

        defaults = {}
        defaults["lr"] = cfg.SOLVER.BASE_LR
        defaults["weight_decay"] = cfg.SOLVER.WEIGHT_DECAY

        norm_module_types = (
            torch.nn.BatchNorm1d,
            torch.nn.BatchNorm2d,
            torch.nn.BatchNorm3d,
            torch.nn.SyncBatchNorm,
            # NaiveSyncBatchNorm inherits from BatchNorm2d
            torch.nn.GroupNorm,
            torch.nn.InstanceNorm1d,
            torch.nn.InstanceNorm2d,
            torch.nn.InstanceNorm3d,
            torch.nn.LayerNorm,
            torch.nn.LocalResponseNorm,
        )

        params: List[Dict[str, Any]] = []
        memo: Set[torch.nn.parameter.Parameter] = set()
        for module_name, module in model.named_modules():
            for module_param_name, value in module.named_parameters(recurse=False):
                if not value.requires_grad:
                    continue
                # Avoid duplicating parameters
                if value in memo:
                    continue
                memo.add(value)

                hyperparams = copy.copy(defaults)
                if "backbone" in module_name:
                    hyperparams["lr"] = hyperparams["lr"] * cfg.SOLVER.BACKBONE_MULTIPLIER
                if (
                    "relative_position_bias_table" in module_param_name
                    or "absolute_pos_embed" in module_param_name
                ):
                    print(module_param_name)
                    hyperparams["weight_decay"] = 0.0
                if isinstance(module, norm_module_types):
                    hyperparams["weight_decay"] = weight_decay_norm
                if isinstance(module, torch.nn.Embedding):
                    hyperparams["weight_decay"] = weight_decay_embed
                params.append({"params": [value], **hyperparams})

        def maybe_add_full_model_gradient_clipping(optim):
            # detectron2 doesn't have full model gradient clipping now
            clip_norm_val = cfg.SOLVER.CLIP_GRADIENTS.CLIP_VALUE
            enable = (
                cfg.SOLVER.CLIP_GRADIENTS.ENABLED
                and cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE == "full_model"
                and clip_norm_val > 0.0
            )

            class FullModelGradientClippingOptimizer(optim):
                def step(self, closure=None):
                    all_params = itertools.chain(*[x["params"] for x in self.param_groups])
                    torch.nn.utils.clip_grad_norm_(all_params, clip_norm_val)
                    super().step(closure=closure)

            return FullModelGradientClippingOptimizer if enable else optim

        optimizer_type = cfg.SOLVER.OPTIMIZER
        if optimizer_type == "SGD":
            optimizer = maybe_add_full_model_gradient_clipping(torch.optim.SGD)(
                params, cfg.SOLVER.BASE_LR, momentum=cfg.SOLVER.MOMENTUM
            )
        elif optimizer_type == "ADAMW":
            optimizer = maybe_add_full_model_gradient_clipping(torch.optim.AdamW)(
                params, cfg.SOLVER.BASE_LR
            )
        else:
            raise NotImplementedError(f"no optimizer type {optimizer_type}")
        if not cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE == "full_model":
            optimizer = maybe_add_gradient_clipping(cfg, optimizer)
        return optimizer

    def configure_optimizers(self):
        optimizer = self.optimizer
        scheduler = self.scheduler
        return [optimizer], [scheduler]

    def forward(self, x):
        logits = self.network(x)
        return logits.argmax(1).squeeze()

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        x = batch
        logits = self.network(x)
        return logits

    def training_step(self, batch_data, batch_idx):
        # optimizers
        optim = self.optimizers()

        # get input & gt_label
        labels = [x["sem_seg_gt"] for x in batch_data]
        labels = torch.stack(labels) 

        # tensorboard logger
        logger = self.logger.experiment

        # get network output
        losses_dict, attention_maps = self.network(batch_data)
        loss = sum(losses_dict.values()) 

        # optimize network
        optim.zero_grad()
        self.manual_backward(loss)
        optim.step()

        # log
        self.log('train/total_loss', loss, sync_dist=True)

        return loss

    def on_train_epoch_end(self):
        scheduler = self.lr_schedulers()
        scheduler.step()

    def validation_step(self, batch_data, batch_idx):
        # get input & gt_label
        images = [x["image"] for x in batch_data]
        labels = [x["sem_seg_gt"] for x in batch_data]

        # tensorboard logger
        logger = self.logger.experiment

        # get network output
        logits, attention_maps = self.network(batch_data)
        logits = [x["sem_seg"] for x in logits]

        images = torch.stack(images) 
        labels = torch.stack(labels) 
        logits = torch.stack(logits) 

        # evaluate performance
        pred  = logits.argmax(1).squeeze().flatten() # prediction and label are both 1-d array, size: minibatch*640*480
        label = labels.squeeze().flatten()

        # update metrics
        self.val_precision.update(pred, label)
        self.val_recall.update(pred, label)
        self.val_iou.update(pred, label)


    def on_validation_epoch_end(self):

        precision = self.val_precision.compute()
        recall = self.val_recall.compute()
        iou = self.val_iou.compute()

        if self.num_classes == 15 :  # For FMB dataset
            ignore_indices = torch.tensor([0, 13]) # ignore unlabeled class and bicycle class (bicycle doesn't appear in test set)
            valid_indices = torch.arange(precision.size(0))

            mask = ~torch.any(valid_indices[:, None] == ignore_indices, dim=1)

            valid_precision = precision[mask]
            valid_recall = recall[mask]
            valid_iou = iou[mask]

            self.log('val/average_precision', valid_precision.mean(), sync_dist=True)  # set sync_dist = True when training via multi-gpu setting.
            self.log('val/average_recall', valid_recall.mean(), sync_dist=True)
            self.log('val/average_IoU', valid_iou.mean(), prog_bar=True, sync_dist=True)
        else:
            self.log('val/average_precision', precision.mean(), sync_dist=True)
            self.log('val/average_recall', recall.mean(), sync_dist=True)
            self.log('val/average_IoU', iou.mean(), prog_bar=True, sync_dist=True)

        assert len(self.label_list) == len(precision), "label_list length must match the number of classes"
        for i in range(len(precision)):
            self.log(f"val(class)/precision_class_{self.label_list[i]}", precision[i].item(), sync_dist=True)
            self.log(f"val(class)/recall_class_{self.label_list[i]}", recall[i].item(), sync_dist=True)
            self.log(f"val(class)/Iou_{self.label_list[i]}", iou[i].item(), sync_dist=True)

        self.val_precision.reset()
        self.val_recall.reset()
        self.val_iou.reset()


    def test_step(self, batch_data, batch_idx):
        images = [x["image"] for x in batch_data]
        labels = [x["sem_seg_gt"] for x in batch_data]

        # get network output
        logits, attention_maps = self.network(batch_data)
        logits = [x["sem_seg"] for x in logits]

        images = torch.stack(images) 
        labels = torch.stack(labels) 
        logits = torch.stack(logits) 

        # evaluate performance
        pred  = logits.argmax(1).squeeze().flatten() 
        label = labels.squeeze().flatten()

        # update metrics
        self.val_precision.update(pred, label)
        self.val_recall.update(pred, label)
        self.val_iou.update(pred, label)

        # save the results
        pred_vis  = visualize_pred(self.palette, logits.argmax(1).squeeze().detach().cpu())
        png_path = os.path.join(self.cfg.SAVE.DIR_ROOT, self.cfg.SAVE.DIR_NAME, "{:05}.png".format(batch_idx))
        cv2.imwrite(png_path, cv2.cvtColor(pred_vis, cv2.COLOR_RGB2BGR))

        # save the attention maps
        if self.cfg.MODEL.SWIN.MODEL_OUTPUT_ATTN:
            attn_dir = os.path.join(self.cfg.SAVE.DIR_ROOT, self.cfg.SAVE.ATTN_DIR_NAME)
            os.makedirs(attn_dir, exist_ok=True)  # Create only the directory structure

            attn_path = os.path.join(attn_dir, "{:05}.npy".format(batch_idx))
            attention_maps = attention_maps.cpu().numpy()
            np.save(attn_path, attention_maps)

        if self.cfg.SAVE.FLAG_VIS_GT:
            # denormalize input images
            images = images.squeeze().detach().cpu().numpy().transpose(1,2,0)
            rgb_vis = images[:,:,:3].astype(np.uint8)
            thr_vis = np.repeat(images[:,:,[-1]], 3, axis=2).astype(np.uint8)
            label_vis = visualize_pred(self.palette, labels.squeeze().detach().cpu())

            png_path = os.path.join(self.cfg.SAVE.DIR_ROOT, "rgb", "{:05}.png".format(batch_idx))
            cv2.imwrite(png_path, cv2.cvtColor(rgb_vis, cv2.COLOR_RGB2BGR))

            png_path = os.path.join(self.cfg.SAVE.DIR_ROOT, "thr", "{:05}.png".format(batch_idx))
            cv2.imwrite(png_path, cv2.cvtColor(thr_vis, cv2.COLOR_RGB2BGR))

            png_path = os.path.join(self.cfg.SAVE.DIR_ROOT, "gt", "{:05}.png".format(batch_idx))
            cv2.imwrite(png_path, cv2.cvtColor(label_vis, cv2.COLOR_RGB2BGR))


    def on_test_epoch_end(self):
        return self.on_validation_epoch_end()