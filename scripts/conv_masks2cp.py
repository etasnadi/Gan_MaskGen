import argparse
from genericpath import exists
from pathlib import Path

import imageio
import matplotlib.pyplot as plt

import cellpose_like_repr

def convert_masks_dir(inp_dir, out_dir):
    out_dir.mkdir(exist_ok=True)

    for inp_im_path in inp_dir.iterdir():
        inp_im = imageio.v2.imread(inp_im_path)
        im_enc = cellpose_like_repr.encode_labels(inp_im)

        imageio.imwrite(out_dir/inp_im_path.name, im_enc[0])

        #im_enc_dec = cellpose_like_repr.reconstruct_mask(im_enc[0])

def get_cli_config():
    parser = argparse.ArgumentParser()

    parser.add_argument('-i', '--input', type=str, default=None, required=True)
    parser.add_argument('-o', '--output', type=str, default=None, required=True)

    return parser.parse_args()

if __name__ == '__main__':
    cli_config = get_cli_config()
    out_dir = Path(cli_config.output)
    out_dir.mkdir(exist_ok=True, parents=True)
    convert_masks_dir(Path(cli_config.input), out_dir)

if __name__ == '__main__' and False:
    
    inp = Path('/media/ervin/Backup/CytoDataset/Timi/collection/dataset/salivary_dry/train/salivary_dry_tiled_masks_aug_256_2')
    outp = Path('/media/ervin/Backup/CytoDataset/Timi/collection/dataset/salivary_dry/train/salivary_dry_tiled_masks_aug_256_2_cp')

    inp = Path('/media/ervin/Backup/TCGA_dataset/val_crops/masks')
    outp = Path('/media/ervin/Backup/TCGA_dataset/val_crops_cp/masks')

    inp = Path('/media/ervin/Backup/CytoDataset/Timi/collection/dataset/melanoma/train_crops/masks')
    outp = Path('/media/ervin/Backup/CytoDataset/Timi/collection/dataset/melanoma/train_crops_cp/masks')

    ###
    
    convert_masks_dir(inp, outp)