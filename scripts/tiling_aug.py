import imageio
import numpy as np
import os
import random
from math import ceil
from skimage import measure, transform
from tqdm import tqdm

verbose = False


def get_tiles(x, k, n_tiles=None, overlap_ratio=.2, permissive=False):
    """ Constructs the tiling parameters in 1D.

    Parameters:
    x (list): the input array size
    k (int): the tile sizes intended to crop from the input
    n_tiles (int)=none: the number of requested tiles
        to use or skipped (should be at least 2 if defined)
    overlap_ratio (float)=.2: if the n_tiles is not specified, it will be determined
        based on the requested overlap ratio between the subsequent input tiles.
    permissive: if True, then leaves the image unmodified it crop does not make sense.

    Returns:
    src_tiles (list((int, int))): what to crop from the source
    src_tile_crops (list(int, int)): crop this from the cropped tile
    taret_tiles (list(int, int)): put the cropped source crop here

    Usage:
    for t in range(len(target_tiles)):
        target[slice(*target_tiles[t])] = source[slice(*src_tiles[t])][slice(*src_tile_crops[t])]
    
    Example:
    >>> get_tiles(100, 25)
    ([(0, 25), (15, 40), (30, 55), (45, 70), (60, 85), (75, 100)], 
     [(0, -5), (5, -5), (5, -5), (5, -5), (5, -5), (5, None)], 
     [(0, 20), (20, 35), (35, 50), (50, 65), (65, 80), (80, 100)])
    
    """

    if k > x or k < 1:
        raise ValueError("The tile size should be greater than 1 and less than or"
                         "equal to the input array size.")

    if n_tiles is not None and (x > n_tiles * k or n_tiles < 2):
        raise ValueError("The sum of tiles should not be less than or equal to the"
                         "input and the number of tiles should be at lest 2.")

    if (permissive == True and n_tiles is None) and k > x * (1 - overlap_ratio):
        # if verbose:
        print('Returning the original image! k=%f x=%f k should be less or equal than: %f' % (
            k, x, x * (1 - overlap_ratio)))
        return [(0, None)], [(0, None)], [(0, None)]

    # Compute the number of tiles automatically.
    # If c is the margin we want to drop, then: x = 2*(k-c)+(n-2)*(k-c). Solving for n gives n=x/(k-2*c)-1 if k > 2.
    if n_tiles is None:
        c = k * overlap_ratio
        n_tiles = max(ceil(x / (k - 2 * c)) - 1, 2)

    # First, determine the centers for each input tile considering the input array coordinates.
    # The first and the last center is in k/2 distance from the sides.
    mid = x - (k / 2) * 2
    step = mid / (n_tiles - 2 + 1)
    pad = k / 2

    tile_centers = [pad] + [pad + (i + 1) * step for i in range(n_tiles - 2)] + [x - pad]

    # Next, determine each tile centered at the previously computed centers.
    src_tiles = [(round(t - k / 2), round(t + (k / 2))) for t in tile_centers]

    # Now, determine the centers (middle points) of the overlaps of the neigbouring tiles.
    overlap_midpts = []
    for i in range(len(src_tiles) - 1):
        next_tile_start, current_tile_end = src_tiles[i + 1][0], src_tiles[i][1]
        midpt = round((next_tile_start + current_tile_end) / 2)
        overlap_midpts.append(midpt)

    # The target tiles are computed by considering the adjacent midpoints as their endpoints.
    target_tiles = []
    for i in range(len(overlap_midpts)):
        prev_midpt = 0 if i == 0 else overlap_midpts[i - 1]
        act_midpt = overlap_midpts[i]
        target_tiles.append((prev_midpt, act_midpt))

    target_tiles += [(overlap_midpts[-1], x)]

    # Determine the croppings from the input to align with the target tiles.
    src_tile_crops = []
    for i in range(len(target_tiles)):
        crop_s = target_tiles[i][0] - src_tiles[i][0]
        crop_e = target_tiles[i][1] - src_tiles[i][1]
        src_tile_crops.append((crop_s, None if crop_e == 0 else crop_e))

    if verbose:
        print('x: %d, #tiles: %s, middle: %d, step: %f' % (x, n_tiles, mid, step))
        print('tile centers:    ', tile_centers)
        print('tiles:           ', src_tiles)
        print('overlap midpts:  ', overlap_midpts)
        print('targets:         ', target_tiles)
        print('src tile crops:  ', src_tile_crops)

    return src_tiles, src_tile_crops, target_tiles


def tiled_copy_nd():
    pass


def test_2D():
    import matplotlib.pyplot as plt
    import imageio

    # im = imageio.imread('/media/ervin/Backup/devel/GANMask/src/prediction_fakes000480.png')[..., 0]
    im = imageio.imread('/home/ervin/Desktop/orban.jpeg')

    plt.imshow(im)

    x = im.shape
    k = (400, 256)
    n_tiles = None

    print(x)

    (yc, _, _), (xc, _, _) = get_tiles(x[0], k[0]), get_tiles(x[1], k[1])

    print(yc)

    fig, axs = plt.subplots(len(yc), len(xc), squeeze=False)
    for y_idx in range(len(yc)):
        for x_idx in range(len(xc)):
            sy = slice(*yc[y_idx])
            sx = slice(*xc[x_idx])
            print('sy, sx', sy, sx)
            f = im[sy, sx, slice(None)]
            axs[y_idx, x_idx].imshow(f)
            print(y_idx, x_idx, sy, sx)

    plt.show()


def test_1D():
    """ Creates a test array a, and a target b with the same size but filled with zeros.
    The task is to copy the contents of a into b using overlapped tiling.
    To make sure the algorithm works it is a good idea to use prime length arrays, 
    request prime number of tiles and prime length tiles to use.

    """
    # length
    x = 587

    # tile size
    k = 20

    # number of tiles
    n_tiles = None

    a = list(range(x))
    b = [0] * len(a)
    print('length of a (source): %d, length of b (target): %d, a==b: %s (before copying)' % (len(a), len(b), a == b))

    src_tiles, src_tile_crops, target_tiles = get_tiles(x, k, n_tiles)

    for i in range(len(target_tiles)):
        t = slice(*target_tiles[i])
        s = slice(*src_tiles[i])
        cr = slice(*src_tile_crops[i])
        if verbose:
            print('Target: %s\t\t\tsource: %s\t\t\tcrop from source: %s' % (t, s, cr))
        b[t] = a[s][cr]

    print('length of a (source): %d, length of b (target): %d, a==b: %s (after copying)' % (len(a), len(b), a == b))


def main():
    test_2D()


if __name__ == '__main__':
    main()


def remove_border_objects(m, dim):
    '''
    Removes the objects that touch the side of the image along the
    requested dimension.
    '''
    a = np.ones_like(m)
    key = [slice(None)] * len(m.shape)
    key[dim] = slice(1, -1)
    a[tuple(key)] = 0

    intersect = a * m

    m = m.copy()

    border_obs = np.unique(intersect)

    for border_ob in border_obs:
        m[m == border_ob] = 0

    return m


def get_random_crop(mask_size, crop_size):
    topleft = [random.randrange(m - c + 1) for m, c in zip(mask_size, crop_size)]
    return [slice(tl, tl + cr) for tl, cr in zip(topleft, crop_size)]


def padim(augmented, target_shape):
    # First pad along the necessary dims
    pad_dims = []
    for dim_idx in range(len(target_shape)):
        augmented_size = augmented.shape[dim_idx]
        target_size = target_shape[dim_idx]

        pad_dim = (0, 0)

        if target_size > augmented_size:
            remove_border_objects(augmented, dim_idx)
            pad_size = target_size - augmented_size
            pad_before = max(pad_size // 2, 0)
            pad_after = max(pad_size - pad_before, 0)
            pad_dim = (pad_before, pad_after)

        pad_dims.append(pad_dim)

    try:
        augmented = np.pad(augmented, tuple(pad_dims), mode='constant')
    except:
        pass
    return augmented


def augment_tile_im(im, m=None, target_shape=(256,256)):
    '''
    Extract the tiles from the image and applies rotation on each.
    If the mask is passed, then the mask is also processed
    '''
    aug_m = []
    aug_im = []

    for k in [0, 1, 2, 3]:
        # Do to the augmentations
        rotated_im = np.rot90(im.copy(), k).astype(im.dtype)
        
        if m is not None:
            rotated_m = np.rot90(m.copy(), k).astype(m.dtype)

        augmented = rotated_im

        if (min(augmented.shape[:2]) > max(target_shape)):
            x = augmented.shape
            k = target_shape
            (yc, _, _) = get_tiles(x[0], k[0], permissive=True)
            (xc, _, _) = get_tiles(x[1], k[1], permissive=True)

            for y_idx in range(len(yc)):
                for x_idx in range(len(xc)):
                    sy = slice(*yc[y_idx])
                    sx = slice(*xc[x_idx])
                    augmented_im_tile = rotated_im.copy()[sy, sx, :]
                    aug_im.append(augmented_im_tile)
                    
                    if m is not None:
                        augmented_m_tile = rotated_m.copy()[sy, sx, :]
                        aug_m.append(augmented_m_tile)
    if m is not None:
        return aug_im, aug_m
    else:
        return aug_im

def augment(m, target_shape, crop=False, tile=True, im=None):
    '''
    If @arg im is not None, then the mask (@arg m) and the image (@arg im)
    will be augmented jointly and both are returned as a pair of augmeted samples.
    '''
    aug = []
    aug_im = []

    print('Extracting tiles')

    for k in [0, 1, 2, 3]:
        # Do to the augmentations
        rotated = np.rot90(m.copy(), k).astype(np.uint16)
        if im is not None:
            rotated_im = np.rot90(im.copy(), k).astype(im.dtype)

        augmented = rotated

        # Guaranteed that the size is at least target_shape
        augmented = padim(augmented, target_shape)

        if tile and (min(augmented.shape[:2]) > max(target_shape)):
            x = augmented.shape
            k = target_shape
            (yc, _, _) = get_tiles(x[0], k[0], permissive=True)
            (xc, _, _) = get_tiles(x[1], k[1], permissive=True)

            n = 0
            # fig, axs = plt.subplots(len(yc), len(xc), squeeze=False)
            for y_idx in range(len(yc)):
                for x_idx in range(len(xc)):
                    n += 1
                    sy = slice(*yc[y_idx])
                    sx = slice(*xc[x_idx])
                    augmented_tile = augmented.copy()[sy, sx]
                    if im is not None:
                        if rotated_im.ndim == 3:
                            augmented_im_tile = rotated_im.copy()[sy, sx, :]
                        else:
                            augmented_im_tile = rotated_im.copy()[sy, sx]

                    n_label_mask = len([u for u in np.unique(m) if u > 0])
                    n_label_crop = len([u for u in np.unique(augmented_tile) if u > 0])

                    if len(np.unique(augmented_tile)) > 3:  # and random.randint(0, 20) < 5
                    #if n_label_mask > 0 and (n_label_crop/n_label_mask) > .1:
                        # axs[y_idx, x_idx].imshow(augmented_tile)

                        cr = get_random_crop(augmented_tile.shape, target_shape)
                        cropped_aug_tile = augmented_tile[tuple(cr)]
                        
                        aug.append(cropped_aug_tile)
                        if im is not None:
                            if augmented_im_tile.ndim == 3:
                                aug_im.append(augmented_im_tile[cr[0], cr[1], :])
                            else:
                                aug_im.append(augmented_im_tile[cr[0], cr[1]])
    return aug, aug_im


def run_agumentation(in_path, out_path, target_shape, target_size=20, norm=True):
    fns = os.listdir(in_path)

    for mask_path in tqdm(fns):
        mask_orig = imageio.imread(os.path.join(in_path, mask_path))

        if len(mask_orig.shape) < len(target_shape):
            _target_shape = target_shape[:len(mask_orig.shape)]
        else:
            _target_shape = target_shape
        if len(mask_orig.shape) == 3 and mask_orig.shape[2] > 3:
            mask_orig = mask_orig[:, :, :3]

        if norm:
            props = measure.regionprops(mask_orig)
            cell_diameters = []
            for prop in props:
                y1, x1, y2, x2 = prop['bbox']
                h = y2 - y1
                w = x2 - x1
                m = .5 * (h + w)
                cell_diameters.append(m)

            med = np.median(cell_diameters)
            scale_factor = target_size / med
            mask_rescaled = transform.rescale(mask_orig, scale_factor, order=0, anti_aliasing=False,
                                              preserve_range=True)

            mask_rescaled = mask_rescaled.astype(np.uint16)
        else:
            mask_rescaled = mask_orig

        mask_augs = augment(mask_rescaled.copy(), _target_shape)
        for aug_id, mask_aug in enumerate(mask_augs):
            # mask_aug_bin = np.stack([(mask_aug > 0).astype(np.uint8) * 255] * 3, -1)
            imageio.imwrite(os.path.join(out_path, f'{" ".join(mask_path.split(".")[:-1])}_{aug_id}.png'), mask_aug)