# Modified by Yan Wang based on the following repositories.
# SimMIM: https://github.com/microsoft/SimMIM/blob/main/data/data_simmim.py
# Mask2Former: https://github.com/facebookresearch/Mask2Former
# CRM_RGBTSeg: https://github.com/UkcheolShin/CRM_RGBTSeg

import numpy as np
from PIL import Image
import torch
import random
import cv2

from fvcore.transforms.transform import Transform

# Modified from Mask2former ColorAugSSDTransform function to take 4 channel input
class ColorAugSSDTransform(Transform):
    """
    A color related data augmentation used in Single Shot Multibox Detector (SSD).

    Wei Liu, Dragomir Anguelov, Dumitru Erhan, Christian Szegedy,
       Scott Reed, Cheng-Yang Fu, Alexander C. Berg.
       SSD: Single Shot MultiBox Detector. ECCV 2016.

    Implementation based on:

     https://github.com/weiliu89/caffe/blob
       /4817bf8b4200b35ada8ed0dc378dceaf38c539e4
       /src/caffe/util/im_transforms.cpp

     https://github.com/chainer/chainercv/blob
       /7159616642e0be7c5b3ef380b848e16b7e99355b/chainercv
       /links/model/ssd/transforms.py
    """

    def __init__(
        self,
        img_format,
        brightness_delta=32,
        contrast_low=0.5,
        contrast_high=1.5,
        saturation_low=0.5,
        saturation_high=1.5,
        hue_delta=18,

        temp_range= (5, 45),
        temp_ratio = 0.2, 
        gradient_shift_ratio = 0.2,
        int_t_ratio = 0.8,
        int_r_ratio = 0.4,

    ):
        super().__init__()
        assert img_format in ["BGR", "RGB", "RGBT"]
        self.is_rgb = img_format == "RGB"
        self.is_rgbt = img_format == "RGBT"
        del img_format
        self._set_attributes(locals())

    def apply_coords(self, coords):
        return coords

    def apply_segmentation(self, segmentation):
        return segmentation

    def apply_image(self, img, interp=None):
        if self.is_rgbt:
            img_r = img[:, :, [2, 1, 0]]
            img_t = img[:, :, -1]
            img_r = self.brightness(img_r)
            if random.randrange(2):
                img_t = self.augment_thermal_image(img_t, img_r) # apply data augmentation for thermal images only
                img_r = self.contrast(img_r)
                img_r = self.saturation(img_r)
                img_r = self.hue(img_r)
                img_t = self.contrast(img_t)
            else:
                img_t = self.augment_thermal_image(img_t, img_r)  # apply data augmentation for thermal images only
                img_r = self.saturation(img_r)
                img_r = self.hue(img_r)
                img_r = self.contrast(img_r)
                img_t = self.contrast(img_t)
            img[:,:,:3]  = img_r[:, :, [2, 1, 0]]
            img[:,:,-1] = img_t
        else:
            if self.is_rgb:
                img = img[:, :, [2, 1, 0]]
            img = self.brightness(img)
            if random.randrange(2):
                img = self.contrast(img)
                img = self.saturation(img)
                img = self.hue(img)
            else:
                img = self.saturation(img)
                img = self.hue(img)
                img = self.contrast(img)
            if self.is_rgb:
                img = img[:, :, [2, 1, 0]]
        return img

    def convert(self, img, alpha=1, beta=0):
        img = img.astype(np.float32) * alpha + beta
        img = np.clip(img, 0, 255)
        return img.astype(np.uint8)

    def brightness(self, img):
        if random.randrange(2):
            return self.convert(
                img, beta=random.uniform(-self.brightness_delta, self.brightness_delta)
            )
        return img

    def contrast(self, img):
        if random.randrange(2):
            return self.convert(img, alpha=random.uniform(self.contrast_low, self.contrast_high))
        return img

    def saturation(self, img):
        if random.randrange(2):
            img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            img[:, :, 1] = self.convert(
                img[:, :, 1], alpha=random.uniform(self.saturation_low, self.saturation_high)
            )
            return cv2.cvtColor(img, cv2.COLOR_HSV2BGR)
        return img

    def hue(self, img):
        if random.randrange(2):
            img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            img[:, :, 0] = (
                img[:, :, 0].astype(int) + random.randint(-self.hue_delta, self.hue_delta)
            ) % 180
            return cv2.cvtColor(img, cv2.COLOR_HSV2BGR)
        return img

    def augment_thermal_image(self, t_img, rgb_img):
        if random.randrange(2):
            # Normalize the image to the specified temperature range
            min_temp, max_temp = self.temp_range
            t_img = t_img.astype(np.float32)
            img_min = np.min(t_img)
            img_max = np.max(t_img)
            # Avoid division by zero by checking if all values are the same
            if img_max - img_min == 0:
                # If all values are the same, set the array to the minimum of the new range
                 t_img = np.full_like(t_img, min_temp)
            else:
                t_img = (t_img - np.min(t_img)) / (np.max(t_img) - np.min(t_img)) * (max_temp - min_temp) + min_temp

            t_img = self.random_temperature_scaling(t_img)
            t_img = self.thermal_gradient_shift(t_img)
            t_img = self.intensity_based_modification(t_img, rgb_img)

            if np.max(t_img) - np.min(t_img) == 0:
                t_img = np.full_like(t_img, 0.)
            else:
                t_img = (t_img - np.min(t_img)) / (np.max(t_img) - np.min(t_img)) * (255. - 0.) + 0.
            return t_img.astype(np.uint8)
        return t_img


    def random_temperature_scaling(self,t_img):
        scale_factor = np.random.uniform(1-self.temp_ratio, 1+self.temp_ratio)  # Scale temperature by ±20%
        return np.clip(t_img * scale_factor, *self.temp_range)

    def thermal_gradient_shift(self,t_img):
        rows, cols = t_img.shape
        gradient = np.linspace(1.0, 1.+self.gradient_shift_ratio, cols).reshape(1, cols)
        return np.clip(t_img* gradient, *self.temp_range)

    def intensity_based_modification(self,t_img, rgb_img):
        # Resize RGB image to match thermal image size
        resized_rgb = cv2.resize(rgb_img, (t_img.shape[1], t_img.shape[0]))
        # Convert the resized RGB image to grayscale
        grayscale = cv2.cvtColor(resized_rgb, cv2.COLOR_RGB2GRAY)
        # Normalize grayscale values and use them to adjust temperature
        normalized_gray = grayscale / 255.0
        return np.clip(t_img * (self.int_t_ratio + self.int_r_ratio * normalized_gray), *self.temp_range)