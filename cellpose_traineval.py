from pathlib import Path
import argparse
from datetime import datetime
import os
import random

import numpy as np
from torch.utils.tensorboard import SummaryWriter
import wandb
import pandas as pd
from stardist.matching import matching_dataset
from tqdm import tqdm
import imageio
import matplotlib.pyplot as plt
from cellpose import models
from cellpose.io import logger_setup

'''
CellPose structure:

UNetModel

Main models:
    CellposeModel -> UnetModel
        Does the actual prediction.

    SizeModel
        Depends on a UnetModel or a CellposeModel to determine the size of the objects in the image.

Cellpose
    Combines CellPoseModel and SizeModel, so the model will be dependent on the object size (only for prediction).

'''

def get_mask_extension(masks_path):
    for f in masks_path.iterdir():
        return f.suffix

def get_dataset(path, lim=None, assume_flows=False):
    print('Processing dataset: %s' % str(path))
    images = []
    masks = []
    image_filenames = []
    mask_filenames = []

    if assume_flows:
        masks_dir = path / 'masks_cp'
    else:
        masks_dir = path / 'masks'

    images_dir = path / 'images'

    print('\timages dir: %s' % str(images_dir))
    print('\tmasks dir: %s' % str(masks_dir))

    mask_ext = get_mask_extension(masks_dir)

    for idx, im_path in enumerate(sorted(list((images_dir).iterdir()))):
        if lim is not None and idx > lim:
            break
        im = imageio.imread(im_path)
        images.append(im)
        mask_path = masks_dir / ('%s%s' % (im_path.stem, mask_ext))
        mask = imageio.imread(mask_path)
        # Mask should be either (y,y) or (3, y, x)
        if mask.ndim > 2:
            mask = np.transpose(mask, (2, 0, 1)).astype(np.float32)
        
        if assume_flows:
            if idx == 0:
                print('Image min/max:', np.min(im), np.max(im), im.shape)
                print('Mask min/max:', np.min(mask), np.max(mask), mask.shape, mask.dtype)
            mask[0, ...] = mask[0, ...]/255.
            mask[1, ...] = (mask[1, ...]-127.)/127.
            mask[2, ...] = (mask[2, ...]-127.)/127.
            if idx == 0:
                print('Image min/max (postprocess):', np.min(im), np.max(im))
                print('Mask min/max (postprocess):', np.min(mask), np.max(mask))

        masks.append( mask )
        image_filenames.append(im_path)
        mask_filenames.append(mask_path)

    print('Number of files: %d' % len(images))

    print('Shape of image/mask #0: %s/%s' % (images[0].shape, masks[0].shape))

    return images, masks, image_filenames, mask_filenames

def merge_datasets(datasets, randomize=False, seed=42):
    from collections import defaultdict
    new_dataset = defaultdict()
    for field_id, _ in enumerate(datasets[0]):
        elems = []
        for ds in datasets:
            elems += ds[field_id]
        if randomize:
            new_dataset[field_id] = random.Random(seed).shuffle(elems)
        else:
            new_dataset[field_id] = elems
    
    merged = tuple(new_dataset[field_id] for field_id in sorted(new_dataset.keys()))
    print('Datasets merged. New size: ', len(merged[0]))
    return merged

def train(datasets, model_path, n_epochs, pretrained_model, train_set, val_set, augment=False, assume_flows=False):
    train_ds = get_dataset(datasets[0] / train_set, assume_flows=assume_flows)
    val_ds = get_dataset(datasets[0] / val_set, assume_flows=assume_flows)

    train_data, train_labels, train_files, _ = train_ds
    test_data, test_labels, test_files, _ = val_ds

    nchan = 3
    channels = [0, 1, 2]
    multichannel = True

    disable_cp_aug = True # Disable cellpose augmentation e.g free rotation, flip
    disable_rescale = True # Disable rescaling to the nuclei instance size

    model = models.CellposeModel(
        gpu=True, model_type=None, nchan=nchan, pretrained_model=pretrained_model)
    model.train(
        train_data=train_data, train_labels=train_labels,
        test_data=test_data, test_labels=test_labels,
        channels=channels, save_every=1, save_path=model_path,
        n_epochs=n_epochs, augment=augment, multichannel=multichannel, disable_cp_aug=disable_cp_aug, disable_rescale=disable_rescale,
        assume_flows=assume_flows)

def eval(dataset, model_path, weight='best', metrics_set=None):
    assert metrics_set is not None, ValueError("Metrics set should be defined!")
    nchan = 3
    normalize = False
    multichannel = True

    model = models.CellposeModel(gpu=True, model_type=None, nchan=nchan, pretrained_model=str(model_path / 'models' / weight))

    test_data, test_labels, test_files, test_mask_files = get_dataset(dataset / metrics_set)

    taus = [.5, .55, .6, .65, .7, .75, .8, .85, .9, .95]

    result_path = model_path / 'eval' / metrics_set / weight
    result_path.mkdir(exist_ok=True, parents=True)

    test_pred = []
    for input_image, labels, im_filename, mask_filename in zip(test_data, test_labels, test_files, test_mask_files):
        pred_masks, flows, styles = model.eval(input_image, diameter=None, channels=[0, 1], normalize=normalize, multichannel=multichannel)
        #import matplotlib.pyplot as plt
        #plt.imshow(masks)
        #plt.show()
        imageio.imwrite(result_path / im_filename.name, input_image)
        imageio.imwrite(result_path / ('%s_pred%s' % (mask_filename.stem, mask_filename.suffix)), pred_masks)
        imageio.imwrite(result_path / ('%s_true%s' % (mask_filename.stem, mask_filename.suffix)), labels)
        test_pred.append(pred_masks)

    stats = [matching_dataset(test_labels, test_pred, thresh=t, show_progress=False) for t in tqdm(taus)]
    eval_df = pd.concat([pd.DataFrame([mi._asdict()]) for mi in stats])

    eval_df.to_csv(result_path / 'stats.csv') 

    return eval_df

def get_config():
    parser = argparse.ArgumentParser()

    parser.add_argument('-ds', '--dataset', type=str, required=True)
    parser.add_argument('--wandb', action='store_true', help='Wandb logging')
    parser.add_argument('--wandb_user', type=str, help='Wandb user.', required=False, default=None)
    parser.add_argument('--wandb_id', type=str, help='Wandb id to continue.', required=False, default=None)
    parser.add_argument('--work_dir', type=str, help='Working directory to save the model into.', required=True)
    parser.add_argument('--model_name', type=str, help='Model name to use on wandb and on the file system.', required=True)
    parser.add_argument('--model_group', type=str, help='Wandb group', required=True)
    parser.add_argument('--eval_weight', type=str, default='best', help='Weight to use when evaluating the model.')
    parser.add_argument('--metrics_set', type=str, default='test_raw', help='Subset in the dataset to use when testing.')
    parser.add_argument('--train_set', type=str, default='train', help='Subset in the dataset to use for training.')
    parser.add_argument('--val_set', type=str, default='val', help='Subset in the dataset to use for validation.')
    parser.add_argument('--resume', type=str, default=False, help='The full path of the weights file to initialize the model.')
    parser.add_argument('--skip_train', action='store_true', help='Skip the training.')
    parser.add_argument('--skip_eval', action='store_true', default=False, help='Skip the evaluation.')
    parser.add_argument('--overwrite', action='store_true', default=False, help='Force overwrite pervious model.')
    parser.add_argument('--augment', action='store_true', default=False, help='Augment on the fly.')
    parser.add_argument('--n_epochs', type=int, default=100, help='Number of epochs')

    # --
    parser.add_argument('--assume_flows', action='store_true', default=False, help='Assume that the input labels are already transformed to flows.')

    return parser.parse_args()

def main():
    logger_setup()

    conf = get_config()

    if conf.wandb:
        print('Wandb logging enabled.')
        assert conf.wandb_user is not None, ValueError("Argument wandb_user is not set.")
        assert conf.wandb_project is not None, ValueError("Argument wandb_project is not set.")
        if conf.wandb_id is not None:
            print('Logging into an existing wandb run: %s' % conf.wandb_id)
            wandb.init(project=conf.wandb_project, entity=conf.wandb_user, sync_tensorboard=False, id=conf.wandb_id)
        else:
            print('Creating a new wandb run.')
            wandb.init(project=conf.wandb_project, entity=conf.wandb_user, group=conf.model_group, name=conf.model_name, sync_tensorboard=False)

    model_path = Path(conf.work_dir) / conf.model_name
    if not conf.skip_train:
        if not model_path.exists() or conf.overwrite:
            print('Augmentation: %s' % conf.augment)
            train(
                [Path(p) for p in conf.dataset.split(',')], 
                model_path, 
                n_epochs=conf.n_epochs, 
                pretrained_model=conf.resume, 
                train_set=conf.train_set, 
                val_set=conf.val_set, 
                augment=conf.augment,
                assume_flows=conf.assume_flows)
        else:
            print('--- Skipping training because the model exists! ---')

    if not conf.skip_eval:
        for act_metrics_set in conf.metrics_set.split(','):
            print('Evaluating on:', act_metrics_set)
            eval_df = eval(Path(conf.dataset), Path(conf.work_dir) / conf.model_name, conf.eval_weight, act_metrics_set)

            print('Result:')
            print(eval_df)

            if conf.wandb:
                # https://docs.wandb.ai/guides/track/log/log-tables
                wandb.log({'Evaluation-%s' % act_metrics_set: wandb.Table(dataframe=eval_df)})

                for i, r in eval_df.iterrows():
                    wandb.log({
                        'threshold': r.thresh,
                        'accuracy': r.accuracy,
                        'precision': r.precision,
                        'recall': r.recall,
                        'f1': r.f1
                    })

                wandb.config.update(conf)

    if conf.wandb:
        wandb.finish()

if __name__ == '__main__':
    main()
