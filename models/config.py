# -*- coding: utf-8 -*-

# Modified by Yan Wang based on the following repositories.
# RTFNet: https://github.com/yuxiangsun/RTFNet
# CRM_RGBTSeg: https://github.com/UkcheolShin/CRM_RGBTSeg


from detectron2.config import CfgNode as CN

def add_peafusion_config(cfg):
    """
    Add config for PEAFusion.
    """

    cfg.MODEL.SWIN.PRETRAINED = None
    cfg.MODEL.SWIN.FROZEN_STAGE = 4
    cfg.MODEL.SWIN.WINDOW_SIZE = 16
    cfg.MODEL.SWIN.PRETRAINED_WINDOW_SIZE = [16,16,16,16]
    cfg.MODEL.SWIN.USE_CHECKPOINT_LIST =  [False,False,False,False]
    # add configuration for adapter
    cfg.MODEL.SWIN.ADD_MODEL_ADAPTER = True
    cfg.MODEL.SWIN.MODEL_ADAPTER_SCALE = 4.0
    cfg.MODEL.SWIN.MODEL_MHA_RATIO = 0.125
    cfg.MODEL.SWIN.MODEL_MHA_GROUPS = 1
    cfg.MODEL.SWIN.MODEL_FFN_RATIO = 0.5
    cfg.MODEL.SWIN.MODEL_OUTPUT_ATTN = True

    cfg.DATASETS.NAME = "MFdataset"
    cfg.DATASETS.DIR = "./datasets/MFdataset/"
    cfg.DATASETS.IMS_PER_BATCH = 8
    cfg.DATASETS.WORKERS_PER_GPU = 4
    cfg.DATASETS.SUB_SAMPLE_RATIO = 1.0

    cfg.SAVE = CN()
    cfg.SAVE.DIR_ROOT = "./results"
    cfg.SAVE.DIR_NAME = "pred"
    cfg.SAVE.ATTN_DIR_NAME = "Attn_maps"
    cfg.SAVE.FLAG_VIS_GT = False
