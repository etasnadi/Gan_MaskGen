import argparse
from calendar import c
import random
from pathlib import Path

import imageio
from stardist import fill_label_holes

import tiling_aug as til

def tiling_aug_images(src, dst, shape, ext='.png'):
    '''
    Only processes the images.
    '''
    for im_path in sorted(list(src.iterdir())):
        im = imageio.imread(im_path)

        # Get all image tiles.
        aug_ims = til.augment_tile_im(im, target_shape=shape)
        dst.mkdir(exist_ok=True)
        for idx, aug_im in enumerate(aug_ims):
            imageio.imwrite(dst/('%s_%d%s' % (im_path.stem, idx, ext)), aug_im)

def tiling_aug_masks(src, dst, shape, src_images=None, dst_images=None, images_ext='.png', log=None):
    '''
    Processes the input masks.
    if @arg src_images is passed, then the corresponding crops are extracted from the images as well (not only the masks).
    '''

    done = []

    print('Log:', log)
    if log is not None:
        f = open(log)
        done = [Path(elem) for elem in f.read().splitlines()]
        print(done)

    for mask_path in sorted(list(src.iterdir())):

        if mask_path in done:
            print('Skipping %s because it is already done.', mask_path)
            continue

        print('Processing', mask_path.name)
        m_raw = imageio.v2.imread(mask_path)
        
        # Skip too big images...
        if m_raw.shape[0] > 2000 or m_raw.shape[1] > 2000:
            continue

        if src_images is not None:
            im = imageio.v2.imread(src_images/('%s%s' % (mask_path.stem, images_ext)))
        
        m = fill_label_holes(m_raw)

        m_augs, im_augs = til.augment(m, shape, im=im)

        print('\t# of tiles: %d' % len(m_augs))

        for aug_idx, (m_aug, im_aug) in enumerate(zip(m_augs, im_augs)):
            if not dst.exists():
                dst.mkdir(parents=True)

            dst_name = dst / ('%s_%d%s' % (mask_path.stem, aug_idx, mask_path.suffix))
            imageio.imwrite(dst_name, m_aug)
            
            if dst_images is not None and not dst_images.exists():
                dst_images.mkdir(parents=True)

            if dst_images is not None:
                dst_name_im = dst_images / ('%s_%d%s' % (mask_path.stem, aug_idx, '.png'))
                imageio.imwrite(dst_name_im, im_aug)

        if log is not None:
            f = open(log, 'a')
            f.write(str(mask_path) + "\n")

def get_cli_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42)

    parser.add_argument('-sm', '--src_masks', type=str, default=None, required=True)
    parser.add_argument('-dm', '--dst_masks', type=str, default=None, required=True)

    parser.add_argument('-si', '--src_images', type=str, default=None, required=True)
    parser.add_argument('-di', '--dst_images', type=str, default=None, required=True)

    parser.add_argument('-sy', '--shape_y', type=int, default=256)
    parser.add_argument('-sx', '--shape_x', type=int, default=256)
    parser.add_argument('-log', '--log', type=str, default=None)

    return parser.parse_args()

if __name__ == '__main__':
    cli_config = get_cli_config()
    random.seed(cli_config.seed)

    shape = (cli_config.shape_y, cli_config.shape_x)
    tiling_aug_masks(
        Path(cli_config.src_masks), 
        Path(cli_config.dst_masks), 
        shape, 
        Path(cli_config.src_images), 
        Path(cli_config.dst_images),
        log=cli_config.log)

