import subprocess
import random
from pathlib import Path
from random import randrange
import argparse

def get_rand_nums(limit, n):
    return ','.join([str(randrange(limit)) for _ in range(n)])

def generate(conf):
    rands = get_rand_nums(conf.rand_lim, conf.n_images)

    stylegan_dir = Path(conf.stylegan_dir)

    #out = work / 'generated' / ('%s' % model.parents[0].name) / model.stem
    #out.mkdir(exist_ok=True, parents=True)

    cmd = [
        'python3', '%s' % str(stylegan_dir/'generate.py'), 
        '--outdir=%s' % str(conf.output), 
        '--seeds=%s' % rands, 
        '--network=%s' % str(conf.model)]

    subprocess.call(cmd)

def generate_labels(conf):
    labels = conf.labels.split(',')
    rands = get_rand_nums(conf.rand_lim, conf.n_images)

    for label in range(labels[0], labels[1]):
        # The actual culster's out dir.
        out = Path(conf.output) / ('%s' % label)
        print('Out dir:', out)
        #out.mkdir(exist_ok=True, parents=True)
        print('Label %s:' % label)
        cmd = [
                'python3', '%s' % str(stylegan_dir/'generate.py'), 
                '--outdir=%s' % str(out), 
                '--seeds=%s' % rands, 
                '--class=%s' % str(label), 
                '--network=%s' % str(conf.model)]

        subprocess.call(cmd)

def main(conf):
    random.seed(conf.seed)

    if conf.labels is not None:
        generate_labels(conf)
    else:
        generate(conf)

def get_cli_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('-code', '--stylegan_dir', type=str, required=True)
    parser.add_argument('-seed', '--seed', type=int, required=False, default=42)
    parser.add_argument('-rl', '--rand_lim', type=int, required=False, default=100000)

    parser.add_argument('-model', '--model', type=str, required=True, help='StyleGAN2-ada model to use for generation.')
    parser.add_argument('-n', '--n_images', type=int, required=False, default=10, help='Number of images to generate.')

    parser.add_argument('-o', '--output', default='stylegan_output', type=str, required=False, help='Output directory to put the results.')
    parser.add_argument('-l', '--labels', type=str, help='Labels to generate (intervals) e.g. 0,12 generates images with labels 0..12.')

    return parser.parse_args()

if __name__ == '__main__':
    cli_config = get_cli_config()
    main(cli_config)
