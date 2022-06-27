import sys
import argparse
import shutil
from pathlib import Path

import imageio

'''

The 'images' dir in the pix2pix test folder:
	seed0074_fake.png (generated image)
	seed0074_real.png (mask)

The StyleGAN generated images dir:
	seed0074.png (StyleGAN mask)

Task: generate the output dir:
	seed0074_fake.png	-> out/images/seed0074.png
	seed0074.png		-> out/images/seed0074.tif

'''

def get_config():
	parser = argparse.ArgumentParser()
	parser.add_argument('-p2p', '--p2p_out', type=str, help='pix2pix generated dataset', required=True)
	parser.add_argument('-out', '--out', type=str, help='collected dataset (result)', required=True)
	parser.add_argument('-fakes', '--fake_masks', type=str, help='fake masks dir', required=True)
	return parser.parse_args()

def copy_all(files, target):
	'''
	Copies the image files.
	The StyleGAN2 filenames are something like: seed090012.png.
	The pix2pix then generates something like seed090012_fake.png.
	We remove the '_real' from the filename before moving it into the final folder.
	'''
	target.mkdir(exist_ok=True, parents=True)
	for file in files:
		desired_fname = file.stem[:-len('_fake')] + file.suffix
		shutil.copy(file, target/desired_fname)

def copy_masks(image_files, target, fake_masks_dir):
	'''
	Copies the masks.
	The pix2pix files are something like seed090012_real.png for the mask output.
	Extract the seed090012 and add the extension (.tif) to the name to get the
	filename of the generated StyleGAN2 mask.
	Then copy the mask from the StyleGAN output folder to the final destination.
	'''
	desired_ext = '.tif'
	target.mkdir(exist_ok=True, parents=True)

	fake_mask_ext = list(fake_masks_dir.iterdir())[0].suffix
	print(fake_mask_ext)

	for file in image_files:
		desired_fname = file.stem[:-len('_real')]
		src_mask = fake_masks_dir/(desired_fname + fake_mask_ext)
		dst_mask = target/(desired_fname + desired_ext)

		if fake_mask_ext != desired_ext:
			imageio.imwrite(dst_mask, imageio.imread(src_mask)[5:-5, 5:-5])
		else:
			shutil.copy(src_mask, dst_mask)

def collect(conf):
	p2p_out = Path(conf.p2p_out)
	out = Path(conf.out)
	fake_masks_dir = Path(conf.fake_masks)
	print('Collecting:', p2p_out, ' -> ', out)

	real_files = list(p2p_out.glob('*_real.png'))
	fake_files = list(p2p_out.glob('*_fake.png'))

	if len(fake_files) != len(real_files):
		print('Error: different number of fakes and reals!')
		sys.exit(1)

	copy_all(fake_files, out/'images')
	copy_masks(real_files, out/'masks', fake_masks_dir)

if __name__ == '__main__':
	_conf = get_config()
	ret = collect(_conf)
