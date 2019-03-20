# import torch
# import torch.autograd
# from torch.autograd import Function, Variable
import numpy as np
from .rank import rank_inf_4d  # , rank_inf
from scipy.ndimage.filters import gaussian_filter  # , median_filter

# 0.9 pour la rankloss + 0.1 l1

# --imsize 64 --sigma 2 --rad 4
# --imsize 128 --sigma 3 --rad 8
# --imsize 256 --sigma 4 --rad 16 (fonctionne pas avec 0.9 pour la rankloss)


# in the whole mini-batch tensor
def rank_inf_4d_test(I, rad):  # 4d
    sh = I.shape
    R = np.zeros(sh)
    # from ipdb import set_trace as st
    # st()
    R_mini = np.zeros(sh) + 1e7
    R_maxi = np.zeros(sh) - 1e7
    for i in range(-rad, rad + 1):  # indice de ligne
        for j in range(-rad, rad + 1):  # indice de colonne
            if i != 0:
                if i < 0:
                    tmp = np.concatenate([I[:, :, -i:, :], np.zeros([sh[0], sh[1], -i, sh[3]])], axis=2)
                else:
                    tmp = np.concatenate([np.zeros([sh[0], sh[1], i, sh[3]]), I[:, :, :-i, :]], axis=2)
            else:
                tmp = I
            if j != 0:
                if j < 0:
                    tmp = np.concatenate([tmp[:, :, :, -j:], np.zeros([sh[0], sh[1], sh[2], -j])], axis=3)
                else:
                    tmp = np.concatenate([np.zeros([sh[0], sh[1], sh[2], j]), tmp[:, :, :, :-j]], axis=3)

            R_maxi[np.logical_and(tmp > 0, R_maxi < tmp)] = tmp[np.logical_and(tmp > 0, R_maxi < tmp)]
            R_mini[np.logical_and(tmp > 0, R_mini > tmp)] = tmp[np.logical_and(tmp > 0, R_mini > tmp)]
            idx = (tmp < I)
            # for m in range(int(math.sqrt((i+j)**2))+1):
            R[idx] = R[idx] + 1

    R_mini[R_mini == 1e7] = 0
    R_maxi[R_maxi == -1e7] = 0
    return R, R_mini, R_maxi


def image_rank(im_ref, rank_neighborhood=5, gaussian_sigma=3):

    im = im_ref.copy()

    # grad = gradient(im, disk(2))

    # s = 5
    # s = 5 + int(im_ref.shape[-1] * 5 / 256)
    # s = int(np.ceil(im_ref.shape[-1] * 10 / 256))

    # f = im_ref.shape[-1]*2/256
    # s = int(1 + im_ref.shape[-1]*16/256)
    # print(im_ref.shape[-1], f, s)
    im = gaussian_filter(im, gaussian_sigma)
    im = im.reshape((1, 1,) + im.shape)
    # im = rank_inf_4d(im, rank_neighborhood)
    im = rank_inf_4d(im, rank_neighborhood)  # modified by mpc
    # im = median_filter(im, 3)

    # im_ref_div = im_ref.copy()
    # im_ref_div[im_ref == 0] = 1
    # im[0, 0] /= im_ref_div

    # im[0, 0] *= im_ref > 0

    # return grad.reshape((1,1)+grad.shape)*im
    return im


def image_rank_4d(im_ref, rank_neighborhood=5, gaussian_sigma=3):

    im = im_ref.copy()

    # f = 1 + im_ref.shape[-1]*8/256
    # s = 2 + im_ref.shape[-1]*4//256

    # f = im_ref.shape[-1]*2/256
    # s = int(im_ref.shape[-1]*16/256)

    # f = 3
    # s = 5
    # print(im_ref.shape[-1], f, s)
    # for each image in the mini-batch, pass a gaussian filter
    for i in range(im_ref.shape[0]):
        im[i, 0] = gaussian_filter(im[i, 0], gaussian_sigma)
    im = rank_inf_4d(im, rank_neighborhood)
    return im
