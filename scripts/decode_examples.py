import os
from functools import partial
from multiprocessing import Pool
from os.path import join as pj
from pathlib import Path

import imageio
import argparse
from tqdm import tqdm

import mask_repr as mr

def get_representations():
    return {
        'gvf': mr.ImgRepresentations.GVF,
        'cp': mr.ImgRepresentations.CP,
    }

def decode_save(fn, input_folder, output_folder, im_loader):
    in_img = Path(input_folder) / fn
    out_img = Path(output_folder) / ('%s.tif' % Path(fn).stem)
    img = im_loader(in_img)
    imageio.imwrite(out_img, img)

def main(config):
    input_folder = config.input
    output_folder = config.output
    img_fomat = get_representations()[config.representation]

    thread_count = config.thread_count

    im_loader = mr.get_img_loader(img_fomat)

    img_fns = os.listdir(input_folder)
    os.makedirs(output_folder, exist_ok=True)

    if thread_count > 1:
        pool = Pool(processes=thread_count)
        pool.map(partial(decode_save, input_folder=input_folder, output_folder=output_folder,
                         im_loader=im_loader), img_fns)

        pool.close()
    else:
        for fn in tqdm(img_fns):
            decode_save(fn, input_folder, output_folder, im_loader)

def get_cli_config():
    parser = argparse.ArgumentParser()

    parser.add_argument('-i', '--input', type=str, default=None, required=True)
    parser.add_argument('-o', '--output', type=str, default=None, required=True)
    parser.add_argument('-rep', '--representation', type=str, choices=list(get_representations().keys()), required=True)
    parser.add_argument('-threads', '--thread_count', type=int, default=6)

    return parser.parse_args()

if __name__ == '__main__':
    config_ = get_cli_config() 
    main(config_)
