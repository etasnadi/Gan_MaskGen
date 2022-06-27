#!/bin/bash

source paths.sh

python3 $P2P_DIR/test.py \
--dataroot $WD/$1 \
--name model_0 \
--model test \
--netG unet_256 \
--direction BtoA \
--dataset_mode single \
--norm batch \
--results_dir $WD/pix2pix_results \
--checkpoints_dir $WD/pix2pix_models \
--num_test $2