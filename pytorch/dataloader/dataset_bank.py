from os import listdir
from os.path import join
from ipdb import set_trace as st
import glob
import sys

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def dataset_std(root, data_split, tasks):
    input_list = sorted(glob.glob(join(root, 'rgb', data_split, '*.png')))
    print(listdir(root))
    st()
    targets_list = []
    for task in tasks:
        targets_list.append(sorted(glob.glob(join(root, task, data_split, '*.png'))))
    # return list(zip(input_list, targets_list))
    return input_list, targets_list

def dataset_target_only(root, phase):
    return sorted(glob.glob(join(root, 'depth', phase, '*.png')))

def dataset_kitti(root, phase, opt):
    phase_path = join(root, phase)
    kitti_split = opt.kitti_split

    # missing_file_list = open('missing_{}_{}'.format(phase, kitti_split), 'w')

    if kitti_split.find('rob') is not -1:
        if phase != 'test':
            if kitti_split.find('_r') is not -1:
                index = '[2]'
            else:
                index = '[2,3]'
            # search for RGB images
            image_search = "2011_*_*_drive_*_sync/image_0{}/*.png".format(index)
            image_files = sorted(glob.glob(join(phase_path, image_search)))

            # search for depth images
            depth_search = "2011_*_*_drive_*_sync/proj_depth/groundtruth/image_0{}/*.png".format(index)
            depth_files = sorted(glob.glob(join(phase_path, depth_search)))
        else:
            # search for RGB images
            image_search = "image/*.png"
            image_files = sorted(glob.glob(join(phase_path, image_search)))

            # limitation of my program
            depth_files = image_files  # correct it to not impose the need of depth images
    elif kitti_split.find('eigen') is not -1:
        image_files = []
        file_ = open("config/kitti/eigen_{}_files_rob.txt".format(phase), "r")

        for f in file_:
            phase_path = phase_path = join(root, phase)
            filenames = f.rsplit(' ', 1)
            if kitti_split.find('_r') is not -1:
                f = filenames[0]
                filepaths = sorted(glob.glob(join(root, '*', f.strip('\n'))))
                if filepaths:
                    image_files.append(filepaths[0])
            else:
                for f in filenames:
                    filepaths = sorted(glob.glob(join(root, '*', f.strip('\n'))))
                    if filepaths:
                        image_files.append(filepaths[0])

        # if phase == 'test':
        #     depth_files = image_files  # correct it to not impose the need of depth images
        # else:
        depth_files = [f.replace('sync/', 'sync/proj_depth/groundtruth/') for f in image_files]
    # st()
    # depth_files = [f.replace('datasets_KITTI', 'datasets_KITTI_2') for f in image_files]
    return image_files, [depth_files]

