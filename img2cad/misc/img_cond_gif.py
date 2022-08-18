"""This script samples from a trained primitive generation model.
"""

import argparse
import os
import json
import time
import enum
from copy import deepcopy
import glob
import multiprocessing as mp
from functools import partial
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from skimage import io
from skimage.metrics import structural_similarity as ssim

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.distributions.categorical import Categorical

import sketchgraphs.data as datalib
from sketchgraphs.data import flat_array
from sketchgraphs.data import Arc, Circle, Line, Point
from sketchgraphs_models.nn.summary import ClassificationSummary
from . import data_utils as i2c_utils
from . import dataset, modules
from .dataset import Token
from .prerender_images import _render_and_save  # TODO: adjust naming


from .train_primitives import device


def _worker(model_path, val_img_path, save_dir, img_idx):
    os.system(R'python -m img2cad.sample_primitives --model_path %s --save_dir %s --top_p 0.9 --num_samples 100 --cond_img_path %s' %
        (model_path, save_dir, val_img_path))


def main():
    parser = argparse.ArgumentParser(
        description='Train primitive generation model')
    parser.add_argument('--saved_models_path', type=str,
        help='Path to saved models')
    parser.add_argument('--save_dir', type=str,
        help='Path to directory for sample saving')
    
    args = parser.parse_args()

    tic = time.time()

    model_paths = sorted(glob.glob(
        os.path.join(args.saved_models_path, '*.pt')))

    val_img_paths = glob.glob('../data/images_val_64/*png')
    val_img_paths = np.random.choice(val_img_paths, size=20)

    for model_idx, model_path in enumerate(model_paths[-1:]):
        workers = []
        for img_idx, val_img_path in enumerate(val_img_paths):
            save_dir = os.path.join(args.save_dir, str(model_idx), str(img_idx))
            p = mp.Process(target=_worker, args=(model_path,
                                                 val_img_path,
                                                 save_dir,
                                                 img_idx))
            p.start()
            workers.append(p)
        [p.join() for p in workers]
        print('\n\n\n\Done with', model_path, '\n\n\n')

    
    print('Took %i seconds' % (time.time() - tic))


if __name__ == '__main__':
    main()