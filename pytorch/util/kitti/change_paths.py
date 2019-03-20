import os
from ipdb import set_trace as st

filename = 'kitti_val_files.txt'
f = open(filename, 'r')
fnew = open('kitti_val_files_rob.txt', 'w')


def change_path(line, index):
    return os.path.join(*(line[index].replace('/data', '').split(os.path.sep)[1:])).replace('.jpg', '.png')


for line in f:
    lines = line.rsplit(' ', 1)
    # print(line2)
    fnew.write('{} {}'.format(change_path(lines, 0), change_path(lines, 1)))
