# Written by Yan Wang based on the following repositories.
# SegMiF: https://github.com/JinyuanLiu-CV/SegMiF
# Mask2Former: https://github.com/facebookresearch/Mask2Former
# CRM_RGBTSeg: https://github.com/UkcheolShin/CRM_RGBTSeg


import cv2
import numpy as np
import os, torch
from imageio import imread
from torch.nn import functional as F
from torch.utils.data.dataset import Dataset
from detectron2.structures import BitMasks, Instances
from detectron2.data import transforms as T

from .augmentation import ColorAugSSDTransform

class FMB_dataset(Dataset):

    def __init__(self, data_dir, cfg, split):
        super(FMB_dataset, self).__init__()

        assert split in ['train', 'val', 'test'], \
            'split must be "train"|"val"|"test"' 

        # Read dataset list, all files have the same name across 'Visible', 'Label', 'Label' folders
        self.data_list = os.listdir(os.path.join(data_dir,split, 'Visible')) 
        self.data_list.sort()

        self.data_dir  = os.path.join(data_dir, split)
        self.split     = split
        self.n_data    = len(self.data_list)
        self.size_divisibility = -1
        self.ignore_label = cfg.MODEL.SEM_SEG_HEAD.IGNORE_VALUE

        self.augmentations = [
            T.ResizeShortestEdge(
                cfg.INPUT.MIN_SIZE_TRAIN,
                cfg.INPUT.MAX_SIZE_TRAIN,
                cfg.INPUT.MIN_SIZE_TRAIN_SAMPLING,
            )
        ]
        if cfg.INPUT.CROP.ENABLED:
            self.augmentations.append(
                T.RandomCrop_CategoryAreaConstraint(
                    cfg.INPUT.CROP.TYPE,
                    cfg.INPUT.CROP.SIZE,
                    cfg.INPUT.CROP.SINGLE_CATEGORY_MAX_AREA,
                    cfg.MODEL.SEM_SEG_HEAD.IGNORE_VALUE,
                )
            )
        if cfg.INPUT.COLOR_AUG_SSD:
            self.augmentations.append(ColorAugSSDTransform(img_format=cfg.INPUT.FORMAT))
        self.augmentations.append(T.RandomFlip())
            
    def read_image(self, name, folder):
        file_path = os.path.join(self.data_dir, '%s/%s' % (folder, name))
        image     = imread(file_path).astype('float32')
        return image

    def __getitem__(self, index):
        name  = self.data_list[index]
        image_rgb = self.read_image(name, 'Visible')
        image_thr = np.expand_dims(self.read_image(name, 'Infrared')[:,:,-1], axis=2) # the infrared image of FMBdataste has three-channel rather than one channel
        image = np.concatenate((image_rgb,image_thr),axis=2)
        # depth = self.read_image(name, 'depth')

        sem_seg_gt = self.read_image(name, 'Label').astype("double")

        # Data Augmentation
        if self.split == 'train':
            aug_input = T.AugInput(image, sem_seg=sem_seg_gt)
            aug_input, transforms = T.apply_transform_gens(self.augmentations, aug_input)
            image = aug_input.image
            sem_seg_gt = aug_input.sem_seg

        # Pad image and segmentation label here!
        image      = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)))
        sem_seg_gt = torch.as_tensor(sem_seg_gt.astype("long"))

        if self.size_divisibility > 0:
            image_size = (image.shape[-2], image.shape[-1])
            padding_size = [
                0,
                self.size_divisibility - image_size[1],
                0,
                self.size_divisibility - image_size[0],
            ]
            image      = F.pad(image, padding_size, value=128).contiguous()
            sem_seg_gt = F.pad(sem_seg_gt, padding_size, value=self.ignore_label).contiguous()

        image_shape = (image.shape[-2], image.shape[-1])  # h, w

        # Packing data
        result = {}
        result["name"]  = name
        result["image"] = image
        result["sem_seg_gt"] = sem_seg_gt.long()

        # # Prepare per-category binary masks
        if sem_seg_gt is not None:
            sem_seg_gt = sem_seg_gt.numpy()
            instances = Instances(image_shape)
            classes = np.unique(sem_seg_gt)
            # remove ignored region
            classes = classes[classes != self.ignore_label]
            instances.gt_classes = torch.tensor(classes, dtype=torch.int64)

            masks = []
            for class_id in classes:
                masks.append(sem_seg_gt == class_id)

            if len(masks) == 0:
                # Some image does not have annotation (all ignored)
                instances.gt_masks = torch.zeros((0, sem_seg_gt.shape[-2], sem_seg_gt.shape[-1]))
            else:
                masks = BitMasks(
                    torch.stack([torch.from_numpy(np.ascontiguousarray(x.copy())) for x in masks])
                )
                instances.gt_masks = masks.tensor

            result["instances"] = instances

        return result

    def __len__(self):
        return self.n_data
