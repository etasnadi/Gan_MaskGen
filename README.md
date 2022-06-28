Generating masks using GAN
===
1. Create a virtualenv; `python3 -m venv venv` and `python3 -m pip install -r requirements.txt`.
1. Load the paths (dataset and working direrctory): `source paths.sh`.
1. Extract tiles (default size: 256x256): ```
python3 scripts/tiling_crop_ds.py
-sm $DS/raw_dataset/masks
-si $DS/raw_dataset/images
-di $WD/crops/images
-dm $WD/crops/masks```

Generating synthetic masks (StyleGAN):

1. Compute the Cellpose representation of the masks: `python3 scripts/conv_masks2cp.py -i $WD/crops/masks -o $WD/crops/masks_cp`.
1. Install StyleGAN from the repository: https://github.com/NVlabs/stylegan2-ada-pytorch.git.
1. Train StyleGAN on the generated masks folder and assume the following results:
    1. `<StyleGAN-train-id>: 00001-masks_cp-auto4-kimg25000-ada-target0.6-bg-resumecustom`
    1. `<StyleGAN-model>: network-snapshot-006249`
1. Generate synthetic masks: ```python3 scripts/genrate_st2ada_cl.py
-code <StyleGAN-dir>
-model $WD/stylegan_models/<StyleGAN-train-id>/<StyleGAN-model>.pkl
-o $WD/synthetic_flows/<StyleGAN-train-id>/<StyleGAN-model>
-n <n-images>```

1. Decode the generated examples: ```python3 scripts/decode_examples.py
-i $WD/synthetic_flows/<StyleGAN-train-id>/<StyleGAN-model>
-o $WD/synthetic_masks/<StyleGAN-train-id>/<StyleGAN-model>
-rep cp```

Generating microscopy iamges for the synthetic masks: (pix2pix):
1. Install pix2pix
    1. git clone https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix.git 
    2. Install in editable mode `python3 -m pip install -e pytorch-CycleGAN-and-pix2pix`. 
    3. Cd to the pix2pix (`$P2P_DIR`) dir and apply the patch `pix2pix_fix.patch` from the repository using `git apply <patch>`.

1. Create the pix2pix training dataset ```
python3 scripts/create_pix2pix_ds.py
--dirA $WD/crops/images
--dirB $WD/crops/masks
--out $WD/pix2pix_train/train```

1. Apply the pix2pix models on the synthetic masks: ```
./pix2pix_test.sh synthetic_masks/<StyleGAN-train-id>/<StyleGAN-model> <n-images>```

2. Collect the synthetic images and masks: ```python3 scripts/collect_pix2pix.py
-p2p $WD/pix2pix_results/model_0/test_latest/images
--out $WD/synthetic_samples
-fakes $WD/synthetic_masks/<StyleGAN-train-id>/<StyleGAN-model>```
