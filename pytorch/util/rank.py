
import numpy as np
import math


def rank_sup(I, rad):
    nl, nc = I.shape
    R = np.zeros([nl, nc])
    for i in range(-rad, rad + 1):  # indice de ligne
        for j in range(-rad, rad + 1):  # indice de colonne
            if i != 0:
                if i < 0:
                     tmp = np.concatenate([I[-i:, :], np.zeros([-i, nc])], axis=0)
                else:
                     tmp = np.concatenate([np.zeros([i, nc]), I[:-i, :]], axis=0)
            else:
                tmp = I
            if j != 0:
                if j < 0:
                    tmp = np.concatenate([tmp[:, -j:], np.zeros([nl, -j])], axis=1)
                else:
                    tmp = np.concatenate([np.zeros([nl, j]), tmp[:, :-j]], axis=1)

            idx = (tmp > I)
            R[idx] = R[idx]+1

    return R


def rank_sup_4d(I, rad):  # 4d
    sh = I.shape
    R = np.zeros(sh)
    for i in range(-rad, rad+1):  # indice de ligne
        for j in range(-rad, rad+1):  # indice de colonne
            if i != 0:
                if i < 0:
                    # print(I[:,:,-i:, :].shape, np.zeros([sh[0], sh[1],-i, sh[3]]).shape )
                    tmp = np.concatenate([I[:,:,-i:, :], np.zeros([sh[0], sh[1],-i, sh[3]])], axis=2)
                else:
                    tmp = np.concatenate([np.zeros([sh[0], sh[1],i, sh[3]]), I[:,:,:-i, :]], axis=2)
            else:
                tmp = I
            if j != 0:
                if j < 0:
                    tmp = np.concatenate([tmp[:,:,:, -j:], np.zeros([sh[0], sh[1],sh[2], -j])], axis=3)
                else:
                    tmp = np.concatenate([np.zeros([sh[0], sh[1],sh[2], j]), tmp[:,:,:, :-j]], axis=3)

            idx = (tmp > I)
            R[idx] = R[idx]+1

    return R


def rank_inf(I, rad):
    nl, nc = I.shape
    R = np.zeros([nl, nc])
    for i in range(-rad, rad+1):
        for j in range(-rad, rad+1):
            if i != 0:
                if i < 0:  # on decalle vers le haut de i lignes
                     tmp = np.concatenate([I[-i:, :], np.zeros([-i, nc])], axis=0)
                else:
                     tmp = np.concatenate([np.zeros([i, nc]), I[:-i, :]], axis=0)
            else:
                tmp = I
            if j != 0:
                if j < 0:
                    tmp = np.concatenate([tmp[:, -j:], np.zeros([nl, -j])], axis=1)
                else:
                    tmp = np.concatenate([np.zeros([nl, j]), tmp[:, :-j]], axis=1)

            idx = (tmp < I)
            R[idx] = R[idx]+1

    return R


def rank_inf_4d(I, rad):  # 4d
    sh = I.shape
    R = np.zeros(sh)
    for i in range(-rad, rad + 1):  # indice de ligne
        for j in range(-rad, rad + 1):  # indice de colonne
            if i != 0:
                if i < 0:
                    # print(I[:,:,-i:, :].shape, np.zeros([sh[0], sh[1],-i, sh[3]]).shape )
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

            idx = (tmp < I)
            # for m in range(int(math.sqrt((i+j)**2))+1):
            R[idx] = R[idx] + 1

    return R
