import argparse
from importlib.metadata import requires
from pathlib import Path

import imageio
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


def create_style_train(A, B, target, mask_channels):
    target.mkdir(exist_ok=True, parents=True)
    As = sorted(list(A.iterdir()))
    Bs = sorted(list(B.iterdir()))
    for imp, mp in tqdm(list(zip(As, Bs))):
        im = imageio.v2.imread(imp)
        m = imageio.v2.imread(mp)

        # Convert the mask to 3 channel image if it is not.
        if mask_channels == 1:
            mRGB = np.stack([m]*3, -1)
        elif mask_channels == 3:
            mRGB = m
        else:
            pass

        res = np.concatenate([im, mRGB], axis=1)
        imageio.imwrite(target/imp.name, res.astype(np.uint8))

def get_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('-dirA', '--dirA', type=str, required=True)
    parser.add_argument('-dirB', '--dirB', type=str, required=True)
    parser.add_argument('-out', '--out', type=str, required=True)
    parser.add_argument('-m_ch', '--mask_channels', type=int, default=1, help='If mask has 1 channel, it will be stacked, if it has 3, nothing will be done.')
    return parser.parse_args()

def main():
    conf = get_config()
    create_style_train(
        Path(conf.dirA), 
        Path(conf.dirB), 
        Path(conf.out),
        conf.mask_channels)

if __name__ == '__main__':
    main()