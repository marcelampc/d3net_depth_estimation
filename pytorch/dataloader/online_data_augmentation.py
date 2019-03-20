import torch.utils.data as data

import os
from os import listdir
from os.path import join
from PIL import Image
from ipdb import set_trace as st
import random
from math import pow
import numpy as np
from random import randint


from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

# global variable to apply crop
state = 0

def rotate_image(img, rotation):
    return img.rotate(rotation, resample=Image.NEAREST)

def scale_image(img, scale, type='distance'):
    img = img.resize((int(img.size[0] * scale), int(img.size[1] * scale)), Image.NEAREST)
    if type == 'distance':
        mode = img.mode
        if img.mode == 'I':
            clip_max = pow(2, 16) - 1
            depth_array = np.array(img)
            depth_array = depth_array.astype(np.float32)
            # depth_array *= ()
            depth_array /= scale
            depth_array = np.clip(depth_array, 0.0, clip_max)
            depth_array = depth_array.astype(np.uint16)
            img = Image.fromarray(depth_array, mode='I;16').convert('I')
        elif img.mode == 'L':
            clip_max = pow(2, 8) - 1
            depth_array = np.array(img)
            depth_array = depth_array.astype(np.float32)
            depth_array /= scale
            depth_array = np.clip(depth_array, 0.0, clip_max)
            depth_array = depth_array.astype(np.uint8)
            img = Image.fromarray(depth_array, mode=mode)
    return img

def crop_image(img, scale, orig_img_size):
    # global state

    borderX = img.size[0] - orig_img_size[0]
    borderY = img.size[1] - orig_img_size[1]

    dX = int(borderX / 2)
    dY = int(borderY / 2)

    random_dX = randint(-dX, dX)
    random_dY = randint(-dY, dY)

    img = img.crop((dX + random_dX,
                    dY + random_dY,
                    dX + random_dX + orig_img_size[0],
                    dY + random_dY + orig_img_size[1]
                    ))

    return img

def color_image(img, R, G, B):
    image_array = np.array(img)

    image_array[:, :, 0] = np.clip(image_array[:, :, 0] * R, 0.0, 255.0)
    image_array[:, :, 1] = np.clip(image_array[:, :, 1] * G, 0.0, 255.0)
    image_array[:, :, 2] = np.clip(image_array[:, :, 2] * B, 0.0, 255.0)

    img = Image.fromarray(image_array)

    return img

def str2bool(values):
    return [v.lower() in ("true", "t") for v in values]


class DataAugmentation():
    def __init__(self, data_augmentation, crop, resize, image_size, scale_to_mm, mean_rotation=0, max_rotation = 5.0, data_transform=None, datatype='distance'):
        self.hflip, self.vflip, self.scale, self.color, self.rotate = str2bool(data_augmentation)

        self.data_transform = data_transform
        self.crop = crop
        self.resize = resize
        self.image_size = image_size
        self.scale_to_mm = scale_to_mm
        self.datatype = datatype
        self.mean_rotation = mean_rotation
        self.max_rotation = max_rotation

        print('Crop: {}'.format(crop))
        print('Resize: {}'.format(resize))
        print('\nData Augmentation')
        print('Hflip: {}'.format(self.hflip))
        print('Vflip: {}'.format(self.vflip))
        print('Scale: {}'.format(self.scale))
        print('Color: {}'.format(self.color))
        print('Rotation: {}'.format(self.rotate))

    def set_probabilities(self):
        self.prob_hflip = random.random()
        self.prob_vflip = random.random()
        self.prob_rotation = np.random.normal(self.mean_rotation, self.max_rotation / 3.0)
        self.prob_scale = np.random.uniform(1.0, 1.5)

        self.random_color_R = np.random.uniform(0.8, 1.2)
        self.random_color_G = np.random.uniform(0.8, 1.2)
        self.random_color_B = np.random.uniform(0.8, 1.2)

    def apply_image_transform(self, *arrays, random_state=None, crop_dims=None):
        import torch
        results = []

        for img in arrays:
            orig_img_size = img.size
            # if random_state:
            #     random.setstate(random_state)
            # Rotate image only if big scale - is it really necessary?
            # DA: Data augmentation
            if self.rotate and self.prob_scale > 1.25:
                img = rotate_image(img, self.prob_rotation)
            # Scale image
            if self.scale:
                img = scale_image(img, self.prob_scale, type=self.datatype)
                img = crop_image(img, self.prob_scale, orig_img_size)
            if self.color and img.mode == 'RGB':
                img = color_image(img, self.random_color_R, self.random_color_G, self.random_color_B)
            if self.hflip and self.prob_hflip < 0.5:
                img = img.transpose(Image.FLIP_LEFT_RIGHT)
            if self.vflip and self.prob_vflip < 0.5:
                img = img.transpose(Image.FLIP_TOP_BOTTOM)
            if self.crop:
                i, j, h, w = self.crop_dims
                img = img.crop((j, i, j + w, i + h))
            if self.resize:
                resize_method = Image.BILINEAR  # Image.ANTIALIAS
                img = img.resize((self.image_size[0], self.image_size[1]), resize_method)
            if img.mode == 'P': # label
                img_tensor = torch.LongTensor(np.array(img, dtype=np.int64))
            else:
                # load depth maps in meters
                if img.mode in ('I', 'L', 'F'):
                    img_tensor = torch.from_numpy(np.array(img)).float()
                    img_tensor = img_tensor.div(self.scale_to_mm).unsqueeze(0) # load in meters
                    # load depth maps normalized:
                # if img.mode == 'I':  # 16 bits
                #     img_tensor = self.data_transform(img)
                #     img_tensor = img_tensor.float().div(pow(2, 16) - 1)
                #     img_tensor = (img_tensor * 2) - 1
                elif img.mode == 'RGB':
                    img_tensor = self.data_transform(img)
                    img_tensor = (img_tensor * 2) - 1
            results.append(img_tensor)

        return results

    def apply_numpy_transform(self, *arrays, flip=True, mirror=True):
        will_flip, will_mirror = False, False
        if self.vflip and random.random() < 0.5:
            will_flip = True
        if self.hflip and random.random() < 0.5:
            will_mirror = True
        
        results = []
        for array in arrays:
            if will_flip:
                if len(array.shape) == 2:
                    array = array[::-1, :]
                else:
                    array = array[:, ::-1, :]
            if will_mirror:
                if len(array.shape) == 2:
                    array = array[:, ::-1]
                else:
                    array = array[:, :, ::-1]
            results.append(np.copy(array))
            
        return tuple(results)
