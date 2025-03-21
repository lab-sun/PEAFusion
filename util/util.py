# By Yuxiang Sun, Dec. 4, 2020
# Email: sun.yuxiang@outlook.com

import numpy as np 
from PIL import Image 
import os

# 0:unlabeled, 1:car, 2:person, 3:bike, 4:curve, 5:car_stop, 6:guardrail, 7:color_cone, 8:bump 
def get_palette_MF():
    unlabelled = [0,0,0]
    car        = [64,0,128]
    person     = [64,64,0]
    bike       = [0,128,192]
    curve      = [0,0,192]
    car_stop   = [128,128,0]
    guardrail  = [64,64,128]
    color_cone = [192,128,128]
    bump       = [192,64,0]
    palette    = np.array([unlabelled,car, person, bike, curve, car_stop, guardrail, color_cone, bump]).astype(np.uint8)
    return palette

# 0:unlabeled, 1:fire_extinhuisher, 2:backpack, 3:hand_drill, 4:rescue_randy
def get_palette_PST():
    unlabelled          = [0,0,0]
    fire_extinhuisher   = [0,0,255]
    backpack            = [0,255,0]
    hand_drill          = [255,0,0]
    rescue_randy        = [255,255,255]
    palette    = np.array([unlabelled, fire_extinhuisher, backpack, hand_drill, rescue_randy]).astype(np.uint8)
    return palette

def get_palette_FMB():
    unlabelled     = [0,   0,   0]
    Road           = [179,228,228]
    Sidewalk       = [181, 57,133]
    Building       = [ 67,162,177]
    Traffic_Lamp   = [200,178, 50]
    Traffic_Sign   = [132, 45,199]
    Vegetation     = [ 66,172, 84]
    Sky            = [179, 73, 79]
    Person         = [ 76, 99,166]
    Car            = [ 66,121,253]
    Truck          = [  6,  6,  6]
    Bus            = [ 12, 12, 12]
    Motorcycle     = [105,153,140]
    Bicycle        = [222,215,158]
    Pole           =  [135,113, 90]
    palette    = np.array([
                            unlabelled, 
                            Road, Sidewalk, Building, Traffic_Lamp, 
                            Traffic_Sign, Vegetation, Sky, Person, Car, 
                            Truck, Bus, Motorcycle, Bicycle, Pole]).astype(np.uint8)
    return palette


def visualize_pred(palette, pred_):
    mapped_label = palette[pred_]
    return mapped_label

def make_save_dir(path_root, pred_name):
    dir_root = os.path.join(path_root)

    if not os.path.exists(dir_root):
        os.mkdir(dir_root)

    dir_pred = os.path.join(path_root, pred_name)
    if not os.path.exists(dir_pred):
        os.mkdir(dir_pred)

    dir_rgb = os.path.join(path_root, 'rgb')
    if not os.path.exists(dir_rgb):
        os.mkdir(dir_rgb)

    dir_thr = os.path.join(path_root, 'thr')
    if not os.path.exists(dir_thr):
        os.mkdir(dir_thr)

    dir_gt  = os.path.join(path_root, 'gt')
    if not os.path.exists(dir_gt):
        os.mkdir(dir_gt)



# We use evalulation metric provided by RTFNet
# Note.
# The unlabeled (or background) class of MF & PST900 dataset is 0. Their evaluations consider the unlabeled label.
# Therefore, the index start from 0.
# The FMB dataset ignores label:0, hence the start_index for FMB should be 1 rather than 0. 
def compute_results(conf_total, ignore_unlabeled = False):
    n_class =  conf_total.shape[0]
    if ignore_unlabeled:
        start_index = 1
    else:
        start_index = 0
    precision_per_class = np.zeros(n_class)
    recall_per_class = np.zeros(n_class)
    iou_per_class = np.zeros(n_class)
    for cid in range(start_index, n_class): # cid: class id
        if conf_total[start_index:, cid].sum() == 0:
            precision_per_class[cid] =  0.
        else:
            precision_per_class[cid] = float(conf_total[cid, cid]) / float(conf_total[start_index:, cid].sum()) # precision = TP/TP+FP

        if conf_total[cid, start_index:].sum() == 0:
            recall_per_class[cid] = 0.
        else:
            recall_per_class[cid] = float(conf_total[cid, cid]) / float(conf_total[cid, start_index:].sum()) # recall = TP/TP+FN

        if (conf_total[cid, start_index:].sum() + conf_total[start_index:, cid].sum() - conf_total[cid, cid]) == 0:
            iou_per_class[cid] = 0.
        else:
            iou_per_class[cid] = float(conf_total[cid, cid]) / float((conf_total[cid, start_index:].sum() + conf_total[start_index:, cid].sum() - conf_total[cid, cid])) # IoU = TP/TP+FP+FN

    return precision_per_class, recall_per_class, iou_per_class
