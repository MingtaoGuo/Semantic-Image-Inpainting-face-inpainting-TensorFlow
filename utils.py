import numpy as np


def get_mask(mask_W, mask_H, img_H, img_W):
    Y = np.random.randint(0, mask_W + 1)
    X = np.random.randint(0, mask_H + 1)
    patch = np.ones([mask_H, mask_W])
    mask = np.zeros([img_H, img_W])
    mask[X:X+mask_H, Y:Y+mask_W] = patch
    return mask, X, Y

def get_patchs(batch, mask_H, mask_W, X, Y):
    return batch[:, X:X+mask_H, Y:Y+mask_W, :]

