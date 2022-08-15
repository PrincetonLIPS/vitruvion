import os
import gzip
import json
import pickle
import time
import numpy as np
import pytest

import torch
from torch.utils.data import DataLoader

from numpy.testing import assert_almost_equal

import sketchgraphs.data as datalib
from img2cad import data_utils as i2c_utils
from img2cad import dataset, constraint_dataset, modules


def test_PrimitiveModel():
    num_bins = 64
    max_len = 100
    max_entities = 16
    batch_size = 4
    embed_dim = 128
    fc_size = 256
    num_heads = 4
    num_layers = 8
    dset = dataset.PrimitiveDataset(
        '../data/sequence_data/sg_t16_validation.npy',
        'img2cad/tests/testdata/images', num_bins, max_len)
    dloader = DataLoader(dset, batch_size, shuffle=True, drop_last=True)
    model = modules.PrimitiveModel(num_bins, 
                                   max_entities, 
                                   embed_dim, 
                                   fc_size, 
                                   num_heads,
                                   num_layers)
    num_val_tokens = len(dataset.Token) + num_bins
    for _, batch in enumerate(dloader):
        output = model(batch)
        assert output.shape == torch.Size([batch_size, max_len, num_val_tokens])


def test_ConstraintModel():
    num_bins = 64
    max_len = 100
    max_entities = 16
    batch_size = 4
    embed_dim = 128
    fc_size = 256
    num_heads = 4
    num_layers = 8
    dset = constraint_dataset.ConstraintDataset(
        '../data/sequence_data/sg_t16_validation.npy',
        'img2cad/tests/testdata/images', num_bins, max_len)
    dloader = DataLoader(dset, batch_size, shuffle=True, drop_last=True)
    model = modules.ConstraintModel(num_bins, 
                                    max_entities, 
                                    embed_dim, 
                                    fc_size, 
                                    num_heads,
                                    num_layers)
    num_val_tokens = len(constraint_dataset.Token) + max_len
    for _, batch in enumerate(dloader):
        output = model(batch)
        assert output.shape == torch.Size([batch_size, max_len, num_val_tokens])