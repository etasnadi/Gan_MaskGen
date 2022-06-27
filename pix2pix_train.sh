#!/bin/bash

source paths.sh

python3 $P2P_DIR/train.py \
	--dataroot $WD/pix2pix_train \
	--name model_0 \
		--model pix2pix \
		--direction BtoA \
		--checkpoints_dir $WD/pix2pix_models \
		--save_epoch_freq 1 \
		--n_epochs 1
