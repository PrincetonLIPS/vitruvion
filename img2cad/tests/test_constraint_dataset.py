import os
import gzip
import json
import pickle
import time

import pytest

import torch
import numpy as np
from numpy.testing import assert_almost_equal

import sketchgraphs.data as datalib
from img2cad import data_utils as i2c_utils
from img2cad import constraint_dataset


def test_tokenize_constraints():
    n_bins = 64
    dset = constraint_dataset.ConstraintDataset(
        '../data/sequence_data/sg_t16_validation.npy',
        'img2cad/tests/testdata/images', n_bins, 100)
    for idx in range(len(dset)):
        sample = dset[idx]