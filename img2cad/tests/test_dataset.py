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
from img2cad import dataset


def test_param_seq_from_tokens():
    n_bins = 64
    # TODO: save small seq array in testdata
    dset = dataset.PrimitiveDataset(
        '../data/sequence_data/sg_t16_validation.npy',
        'img2cad/tests/testdata/images', n_bins, 100)
    for idx in range(len(dset)):
        seq = dset.get_seq(idx)
        sample = dset[idx]
        sketch = datalib.sketch_from_sequence(seq)
        i2c_utils.normalize_sketch(sketch) 
        ori_param_bins = [
            i2c_utils.quantize_params(
                i2c_utils.parameterize_entity(
                    ent), n_bins) for ent in sketch.entities.values()]
        recon_param_bins = dataset.param_seq_from_tokens(sample['val'])
        assert np.array_equal(np.concatenate(ori_param_bins),
                              np.concatenate(recon_param_bins))


def test_sketch_from_tokens():
    n_bins = 64
    dset = dataset.PrimitiveDataset(
        '../data/sequence_data/sg_t16_validation.npy',
        'img2cad/tests/testdata/images', n_bins, 100)
    for idx in range(len(dset)):
        seq = dset.get_seq(idx)
        sketch = datalib.sketch_from_sequence(seq)
        i2c_utils.normalize_sketch(sketch) 
        sample = dset[idx]
        sketch2 = dataset.sketch_from_tokens(sample['val'], n_bins)
        for ent, ent2 in zip(sketch.entities.values(),
                             sketch2.entities.values()):
            if ent2 is None:
                if isinstance(ent, datalib.Arc):
                    # Skip arc that collapsed due to binning
                    continue
            if isinstance(ent, datalib.Line):
                if ent2.startParam == ent2.endParam:
                    # Skip line that collapsed due to binning
                    # TODO: address bin collapse issue
                    continue
            param_bins = i2c_utils.quantize_params(
                i2c_utils.parameterize_entity(ent), n_bins)
            param_bins2 = i2c_utils.quantize_params(
                i2c_utils.parameterize_entity(ent2), n_bins)
            assert (np.abs(param_bins - param_bins2) <= 1).all()
            # TODO: address imprecise quantization issue.
            # Allowing binning diff of 1 for now


def test_dset_use_images():
    n_bins = 64
    dset = dataset.PrimitiveDataset(
        '../data/sequence_data/sg_t16_validation.npy',
        'img2cad/tests/testdata/images', n_bins, 100, use_images=True)
    for sample in dset:
        img = sample['img']
        assert img.dim() == 3
        assert img.shape[0] == 1
        assert (img <= 1).all() and (img >= -1).all()