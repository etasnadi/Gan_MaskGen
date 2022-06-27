import numpy as np


def quantize(dP, mag=5.):
    min_ = -1. * mag
    max_ = 1. * mag

    dP -= min_
    dP /= max_ - min_
    return (dP * 255.).astype(np.uint8)


def dequantize(dP, mag=5.):
    min_ = -1. * mag
    max_ = 1. * mag

    dP = dP.astype(np.float32) / 255.
    dP *= (max_ - min_)
    dP += min_
    return dP
