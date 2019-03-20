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

from .dataset_utils import get_paths_list, load_img, get_params, check_files

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from .online_data_augmentation import DataAugmentation

# global variable to apply crop
random_state = 0

class DatasetFromFolder(data.Dataset):
    def __init__(self, opt, root, phase, data_split, data_augmentation, crop=True, resize=True, data_transform=None, imageSize=[256], outputSize=0, dataset_name='nyu'):
        super(DatasetFromFolder, self).__init__()
        self.input_list, self.target_list = get_paths_list(root, opt.dataset_name, phase, data_split, opt.tasks, opt)

        check_files(self.input_list, root)
        [check_files(self.target_list[idx], root) for idx in range(len(self.target_list))]

        # here, we have a list with gt, file.png... and label, file.png...
        self.data_transform = data_transform
        # self.dataset_files_list = dataset_files_list
        self.imageSize = imageSize if len(imageSize) == 2 else imageSize * 2
        self.dataset_name = dataset_name
        self.scale_to_mm = opt.scale_to_mm # scale data to millimeters

        if outputSize == 0:
            self.outputSize = self.imageSize
        else:
            self.outputSize = outputSize if len(outputSize) == 2 else outputSize * 2
        print(self.outputSize)

        # self.DA_hflip, self.DA_scale, self.DA_color, self.DA_rotate = str2bool(data_augmentation)
        self.data_augmentation = data_augmentation
        self.crop = crop
        self.resize = resize
        self.state = 0
        self.data_augm_obj = DataAugmentation(data_augmentation, crop, resize,
                                                self.imageSize, opt.scale_to_mm, 
                                                data_transform=self.data_transform,
                                                mean_rotation=0, max_rotation = 5.0,)

    def __getitem__(self, index):
        input_img = load_img(self.input_list[index])[0]
        target_imgs = [load_img(target[index])[0] for target in self.target_list]

        if self.dataset_name == 'kitti':
            from .dataset_utils import crop_kitti_supervised
            input_img = crop_kitti_supervised(input_img)
            target_imgs = [crop_kitti_supervised(target_imgs[0])]

        if self.crop:
            crop_dims = get_params(target_imgs[0], crop_size=[self.imageSize[0], self.imageSize[1]])
        else:
            crop_dims = 0, 0, 0, 0

        self.data_augm_obj.set_probabilities()
        self.data_augm_obj.crop_dims = crop_dims
        # input_img_tensor, img_target_tensor = self.data_augm_obj.apply_image_transform(input_img, img_target, random_state=random_state, crop_dims=crop_dims)

        input_img_tensor = self.data_augm_obj.apply_image_transform(input_img)[0]
        
        targets_tensor = [self.data_augm_obj.apply_image_transform(target)[0] for target in target_imgs]

        return input_img_tensor, targets_tensor

    def __len__(self):
        return 1 #len(self.input_list)
