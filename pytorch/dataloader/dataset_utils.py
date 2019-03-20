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
from .online_data_augmentation import DataAugmentation

# global variable to apply crop
random_state = 0

def check_files(files_list, root):
    if len(files_list) == 0:
        from .dataset_bank import IMG_EXTENSIONS
        raise(RuntimeError("Found 0 images in subfolders of: " + root + "\n"
                            "Supported image extensions are: " + ",".join(IMG_EXTENSIONS)))
    else:
        print("Seems like your path is ok! =) I found {} images!".format(len(files_list)))

def get_paths_list(root, dataset_name, phase, data_split, tasks, opt=None):
    if dataset_name == 'nyu' or dataset_name == 'make3D':
        from .dataset_bank import dataset_std
        return dataset_std(root, data_split, tasks)
    if dataset_name == 'kitti':
        from .dataset_bank import dataset_kitti
        return dataset_kitti(root, data_split, opt)

def load_img(*filepaths):
    paths = []
    for path in filepaths:
        paths.append(Image.open(path))
    return paths

def crop_kitti_supervised(img):
    # resize because idk why dataset images do not have the same resolution
    img = img.resize((1242, 375), Image.NEAREST)
    # crop 1/3 of the image like Eigen et al.
    height_start = img.size[1] // 3.2  # did this to have an image with height > 256
    img = img.crop((0,
                    height_start,
                    img.size[0],
                    img.size[1]
                    ))
    return img

def get_params(img_input, crop_size=[224, 224]):
    w, h = img_input.size
    tw, th = crop_size

    if w == tw and h == th:
        return 0, 0, h, w

    i = random.randint(0, h - th)
    j = random.randint(0, w - tw)
    return i, j, th, tw