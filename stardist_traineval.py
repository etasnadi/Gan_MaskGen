from __future__ import print_function, unicode_literals, absolute_import, division

import argparse
import itertools
import os
import sys
from datetime import datetime
from glob import glob

import numpy as np
import pandas as pd
import imageio
import tensorflow as tf
import wandb
from csbdeep.utils import Path, normalize
from stardist import fill_label_holes, calculate_extents, gputools_available
from stardist.matching import matching_dataset
from stardist.models import Config2D, StarDist2D
from tqdm import tqdm

from data_aug.utils.stardist_aug import AUGMENTS, get_augmenter


def get_dataset(dataset_subdir, return_filenames=False):
    print('Reading dataset:', dataset_subdir)

    X_files = list((dataset_subdir / 'images/').glob('*.png'))
    Y_files = []
    for x_file in X_files:
        Y_files.append(dataset_subdir / 'masks/' / ('%s.tif' % x_file.stem))

    print_first_n = 20

    print('First %d elements' % print_first_n)
    for idx, (x, y) in enumerate(zip(X_files, Y_files)):
        if idx < print_first_n:
            print(Path(x).name, Path(y).name)

    remaining = len(X_files) - print_first_n

    if remaining > 0:
        print('... %d more' % remaining)

    # assert all(Path(x).stem == Path(y).stem for x, y in zip(X_files, Y_files))

    X = list(map(imageio.imread, X_files))
    Y = list(map(imageio.imread, Y_files))

    def convert_RGB(X):
        '''
        Converts the input to RGB (h, w, 3).

        Possible inputs:
        (h, w)      =>  add the last channel and repeat 3x
        (h, w, 1)   =>  repeat the channel 3x
        (h, w, 3)   =>  does noting
        (h, w, 4)   =>  drops the last channel
        otherwise fail.
        '''
        X_color = []
        for x_raw in X:
            if x_raw.ndim == 2:
                X_color.append(np.stack([x_raw] * 3, -1))
            elif x_raw.ndim == 3 and x_raw.shape[-1] == 1:
                X_color.append(np.concatenate([x_raw] * 3, -1))
            elif x_raw.ndim == 3 and x_raw.shape[-1] == 4:
                X_color.append(x_raw[..., :3])
            elif x_raw.ndim == 3 and x_raw.shape[-1] == 3:
                X_color.append(x_raw)  # The input format is already RGB.
            else:
                print("Error: can't convert the input to RGB! Input shape:", x_raw.shape)
                #sys.exit(1)
        return X_color

    print('Converting input to RGB...')
    X = convert_RGB(X)

    n_channel = 1 if X[0].ndim == 2 else X[0].shape[-1]

    print('Network input number of channels:', n_channel)

    axis_norm = (0, 1)  # normalize channels independently
    # axis_norm = (0,1,2) # normalize channels jointly
    if n_channel > 1:
        print(
            "Normalizing image channels %s." % ('jointly' if axis_norm is None or 2 in axis_norm else 'independently'))
        sys.stdout.flush()

    X = [normalize(x, 1, 99.8, axis=axis_norm) for x in tqdm(X)]
    Y = [fill_label_holes(y) for y in tqdm(Y)]

    if return_filenames:
        return X, Y, X_files, Y_files
    else:
        return X, Y


def eval(conf, metrics_set):
    '''
    @arg congf(
        dataset
        metrics_set
        work_dir
        model_name
    )
    '''

    print('Evaluating on: %s...' % metrics_set)

    X_val, Y_val, X_files, Y_files = get_dataset(Path(conf.dataset)/metrics_set, return_filenames=True)

    model = StarDist2D(None, name=conf.model_name, basedir=str(conf.work_dir))

    taus = [.5, .55, .6, .65, .7, .75, .8, .85, .9, .95]

    model_path = Path(conf.work_dir) / conf.model_name
    result_path = model_path / 'eval' / metrics_set / 'best'
    result_path.mkdir(exist_ok=True, parents=True)

    Y_val_pred = []
    for x, y, x_file, y_file in zip(X_val, Y_val, X_files, Y_files):
        y_pred = model.predict_instances(x, n_tiles=model._guess_n_tiles(x), show_tile_progress=False)[0]
        imageio.imwrite(result_path / x_file.name, x)
        imageio.imwrite(result_path / ('%s_pred%s' % (y_file.stem, y_file.suffix)), y_pred.astype(np.uint16))
        imageio.imwrite(result_path / ('%s_true%s' % (y_file.stem, y_file.suffix)), y)
        Y_val_pred.append(y_pred)

    stats = [matching_dataset(Y_val, Y_val_pred, thresh=t, show_progress=False) for t in tqdm(taus)]
    eval_df = pd.concat([pd.DataFrame([mi._asdict()]) for mi in stats])
    
    eval_df.to_csv(result_path/'stats.csv')

    print('Evaluation done:')
    print(eval_df)

    if conf.wandb:
        wandb.log({
            'Evaluation-%s' % metrics_set: wandb.Table(dataframe=eval_df)
        })

        wandb.log({'metrics_set': metrics_set})
        for i, r in eval_df.iterrows():
            wandb.log({
                'threshold': r.thresh,
                'accuracy': r.accuracy,
                'precision': r.precision,
                'recall': r.recall,
                'f1': r.f1
            })


def train(conf):
    '''
    @arg conf(
        dataset
        train_set
        val_set
        model_name
        work_dir
        resume
        preferred_weight
    )
    '''
    X_trn, Y_trn = get_dataset(Path(conf.dataset) / conf.train_set)
    X_val, Y_val = get_dataset(Path(conf.dataset) / conf.val_set)
    
    # print('number of images: %3d' % len(X))
    print('- training:       %3d' % len(X_trn))
    print('- validation:     %3d' % len(X_val))

    # Use OpenCL-based computations for data generator during training (requires 'gputools')
    use_gpu = False and gputools_available()

    sd_conf = get_sd_config(conf, X_trn, use_gpu)

    if use_gpu:
        from csbdeep.utils.tf import limit_gpu_memory
        # adjust as necessary: limit GPU memory to be used by TensorFlow to leave some to OpenCL-based computations
        limit_gpu_memory(0.8)
        # alternatively, try this:
        # limit_gpu_memory(None, allow_growth=True)

    if conf.resume is not None:
        path = Path(conf.resume)
        model = StarDist2D(config=None, name=path.stem, basedir=path.parent)
        model.config = sd_conf
        model.name = Path(str(conf.model_name))
        model.basedir = Path(str(conf.work_dir))
        pref = conf.preferred_weight
        model._find_and_load_weights(prefer=pref)
        model._set_logdir()
    else:
        model = StarDist2D(sd_conf, name=str(conf.model_name), basedir=str(conf.work_dir))

    '''
    # Disable layers
    for l in model.keras_model.layers:
        if l.name.startswith('up'):
            l.trainable = True
        else:
            l.trainable = False

    model.keras_model.get_layer('dist').trainable = True
    model.keras_model.get_layer('prob').trainable = True
    model.keras_model.get_layer('features').trainable = True
    # ###
    '''

    median_size = calculate_extents(list(Y_trn), np.median)
    fov = np.array(model._axes_tile_overlap('YX'))
    print(f"median object size:      {median_size}")
    print(f"network field of view :  {fov}")
    if any(median_size > fov):
        print("WARNING: median object size larger than field of view of the neural network.")

    augmenter = get_augmenter(conf)

    if not conf.skip_train:
        model.train(X_trn, Y_trn,
                    validation_data=(X_val, Y_val), augmenter=augmenter, epochs=conf.n_epochs,
                    steps_per_epoch=len(Y_trn))

    if not conf.skip_th:
        model.optimize_thresholds(X_val, Y_val)


def get_sd_config(conf, X_trn, use_gpu):
    print(Config2D.__doc__)

    n_channel = 1 if X_trn[0].ndim == 2 else X_trn[0].shape[-1]
    sd_conf = Config2D(
        n_rays=conf.n_rays,
        grid=conf.grid,
        use_gpu=use_gpu,
        n_channel_in=n_channel,
        train_n_val_patches=conf.n_val_patches,
        train_learning_rate=conf.lr,

        # Set train loss weight...
        #train_loss_weights=(2., 0.4),
    )

    if conf.disable_reduce_lr >= 0:
        sd_conf.train_reduce_lr = None

    print(sd_conf)
    vars(sd_conf)

    return sd_conf


def get_config():
    aug_pipes = [''.join(x) for x in itertools.chain.from_iterable([
        itertools.combinations(AUGMENTS, i) for i in range(2, len(AUGMENTS) + 1)
    ])]

    parser = argparse.ArgumentParser()

    parser.add_argument('-ds', '--dataset', type=str, default=None)
    parser.add_argument('--wandb', action='store_true', help='Wandb logging')
    parser.add_argument('--wandb_user', type=str, help='Wandb username', required=False, default=None)
    parser.add_argument('--wandb_project', type=str, help='Wandb project', required=False, default=None)
    parser.add_argument('--wandb_id', type=str, help='Wandb id to continue.', required=False, default=None)
    parser.add_argument('--work_dir', type=str, default='models')
    parser.add_argument('--model_name', type=str)
    parser.add_argument('--model_group', type=str)
    parser.add_argument('--eval_weight', type=str, default='best', help='Weight to use when evaluating the model.')
    parser.add_argument('--metrics_set', type=str, default='test', help='Which subset is used for computing the metrics.')
    parser.add_argument('--train_set', type=str, default='train', help='Which subset is used for training.')
    parser.add_argument('--val_set', type=str, default='val', help='Which subset is used for validation.')
    parser.add_argument('--resume', type=str, default=None, help='location of model to load')
    parser.add_argument('--skip_train', action='store_true', help='Skip training, using for threshold optimization')
    parser.add_argument('--skip_eval', action='store_true', default=False, help='Skip the evaluation.')
    # skip eval option should be added
    # overwrite option should be added
    parser.add_argument('--aug_pipe', choices=aug_pipes, default='',
                            help='f=Flips, i=Intensity changes, g=Gaussian noise')
    parser.add_argument('--n_epochs', type=int, default=200)
    # ---
    parser.add_argument('--preferred_weight', type=str, default='best')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('-r', '--time_to_repeat', type=int, default=1)
    parser.add_argument('--lr', type=float, default=0.003, help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=4, help='Train batch size')
    parser.add_argument('--n_val_patches', type=int, default=None, help='Number of patches extracted from each validation image')
    parser.add_argument('--n_rays', type=int, default=32,
                        help='32 is a good default choice (see 1_data.ipynb)')
    parser.add_argument('--grid', type=tuple, default=(2, 2),
                        help='Predict on subsampled grid for increased efficiency and larger field of view')
    parser.add_argument('--skip_th', action='store_true', help='Skip threshold optimization', default=False)
    parser.add_argument('--skip_model_naming', action='store_true', help='Use exact model names')
    parser.add_argument('--device', type=int, default=-1, help='CUDA device')
    parser.add_argument('--disable_reduce_lr', type=int, default=-1, help='Disable reduce LR on plateau.')

    #parser.add_argument('--evaluate', type=bool, default=False, help='Evaluate or train a model.')

    return parser.parse_args()

def main():
    conf = get_config()

    if conf.wandb:
        assert conf.wandb_user != None and conf.wandb_project != None, ValueError("Wandb user and project should be defined if wandb is enabled!")
        print('Wandb logging enabled.')
        if conf.wandb_id is not None:
            print('Logging into an existing wandb run: %s' % conf.wandb_id)
            wandb.init(project=conf.wandb_project, entity=conf.wandb_user, id=conf.wandb_id, sync_tensorboard=False)
        else:
            print('Creating a new wandb run.')
            wandb.init(project=conf.wandb_project, entity=conf.wandb_user, group=conf.model_group, name=conf.model_name, sync_tensorboard=False)

    model_path = Path(conf.work_dir) / conf.model_name
    if not conf.skip_train:
        if not model_path.exists() or conf.overwrite:
            print('Augmentation: %s' % conf.aug_pipe)
            train(conf)
        else:
            print('--- Skipping training because the model exists! ---')

    if not conf.skip_eval:
        for metrics_set in conf.metrics_set.split(','):
            eval(conf, metrics_set)


    if conf.wandb:
        wandb.config.update(conf)
        wandb.finish()


if __name__ == '__main__':
    main()
