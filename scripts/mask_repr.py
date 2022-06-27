import enum

import imageio
import numpy as np
from skimage import io as sk_io

import cellpose_like_repr as cp
#import data_aug.mask_gen.mask_represantation.gvf_repr as gvf
#import data_aug.mask_gen.mask_represantation.recolor as rc


class ImgRepresentations(enum.Enum):
    C_LABEL = 'clabel'
    RC = 'rc'
    CP = 'cp'
    SKE_HEAT = 'ske_heat'
    GVF = 'gvf'
    RGB = 'rgb'


def get_img_loader(img_format: ImgRepresentations):
    """
    Get image decoder function for given image format
    :param img_format: One of ImgRepresentations (IR)
    :return:
    """
    if img_format == ImgRepresentations.C_LABEL:
        return load_gclabeled_img
    elif img_format == ImgRepresentations.RC:
        return load_colored_masks
    elif img_format == ImgRepresentations.CP:
        return load_cp_repr
    elif img_format == ImgRepresentations.SKE_HEAT:
        return load_ske_heat
    elif img_format == ImgRepresentations.GVF:
        return load_gvf
    elif img_format == ImgRepresentations.RGB:
        return load_rgb
    else:
        raise ValueError('Invalid img format')


def get_img_encoder(img_format: ImgRepresentations):
    if img_format == ImgRepresentations.CP:
        return encode_cp
    if img_format == ImgRepresentations.SKE_HEAT:
        return encode_ske_heat
    if img_format == ImgRepresentations.GVF:
        return encode_gvf
    if img_format == ImgRepresentations.RC:
        return encode_recolored


def load_gclabeled_img_to_bin(fn):
    img = sk_io.imread(fn)
    img_th = np.zeros(img.shape).astype(np.uint8)
    img_th[np.nonzero(img)] = 255

    return img_th


def load_colored_masks_to_bin(fn):
    from data_aug.mask_gen.mask_represantation.reconstract_labeling import re_labeling
    img = sk_io.imread(fn)

    labeled_mask, im_white, im_c = re_labeling(img)

    return im_white


def load_gclabeled_img(fn):
    img = sk_io.imread(fn)
    # img_th = np.zeros(img.shape).astype(np.uint8)
    # img_th[np.nonzero(img)] = 255

    return img


def load_rgb(fn):
    img = sk_io.imread(fn)
    return img


def encode_recolored(fn, **kwargs):
    im = imageio.imread(fn)

    # set defaults
    if 'colors' not in kwargs:
        kwargs['colors'] = [
            (255, 0, 0),
            (0, 255, 0),
            (0, 0, 255),
            (255, 255, 0),
        ]
    if 'to_rgb' not in kwargs:
        kwargs['to_rgb'] = True
    if 'base_color' not in kwargs:
        kwargs['base_color'] = (255, 255, 255)

    return rc.re_color_im(im, **kwargs)


def load_colored_masks(fn):
    from data_aug.mask_gen.mask_represantation.reconstract_labeling import re_labeling
    img = sk_io.imread(fn)

    labeled_mask, im_white, im_c = re_labeling(img)

    return labeled_mask


def th_cell_prob(img, th=220):
    # thresholding cell probability layer
    cell_prob = np.zeros(img[:, :, 0].shape)
    points = np.where(img[:, :, 0] > th)
    cell_prob[points] = 255
    img_p = np.stack([cell_prob, img[:, :, 1], img[:, :, 2]]).transpose((1, 2, 0)).astype(np.uint8)

    return img_p


def encode_cp(fn):
    im = imageio.imread(fn)
    enc_im, dP = cp.encode_labels(im, method='cp')
    return enc_im


def load_cp_repr(fn):
    from cellpose_like_repr import reconstruct_mask
    img = sk_io.imread(fn)

    img_p = th_cell_prob(img)
    mask, p = reconstruct_mask(img_p)

    # TODO postprocess?
    return mask


def encode_ske_heat(fn):
    im = imageio.imread(fn)
    enc_im, dP = cp.encode_labels(im, method='ske_heat')
    return enc_im


def load_ske_heat(fn):
    return load_cp_repr(fn)


def encode_gvf(fn):
    im = imageio.imread(fn)
    gvf_im, gvf_y, gvf_x = gvf.encode_img(im)
    return gvf_im


def load_gvf(fn):
    img = sk_io.imread(fn)
    img_p = th_cell_prob(img, 50)

    res_im, _, _ = gvf.decode_img(img_p)

    return res_im
