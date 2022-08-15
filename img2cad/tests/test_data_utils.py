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
from sketchgraphs.data import EntityType
from img2cad import data_utils as i2c_utils
from img2cad import noise_models


def test_get_sketch_bbox(sketches):
    # TODO: run a few manual test cases
    for sketch in sketches:
        (x0, y0), (x1, y1) = i2c_utils.get_sketch_bbox(sketch)
        assert x1 >= x0 and y1 >= y0


def test_center_sketch(sketches):
    for sketch in sketches:
        i2c_utils.center_sketch(sketch)
        (x0, y0), (x1, y1) = i2c_utils.get_sketch_bbox(sketch)
        assert_almost_equal(np.mean([x0, x1]), 0)
        assert_almost_equal(np.mean([y0, y1]), 0)


def test_rescale_sketch(sketches):
    for sketch in sketches:
        scale_factor = i2c_utils.normalize_sketch(sketch)
        if scale_factor == -1:  # zero-dim sketch
            continue
        (x0, y0), (x1, y1) = i2c_utils.get_sketch_bbox(sketch)
        w = x1 - x0
        h = y1 - y0
        assert_almost_equal(max(w, h), 1)


def test_entity_parameterization_roundtrip(sketches):
    for sketch in sketches:
        for ent in sketch.entities.values():
            params = i2c_utils.parameterize_entity(ent)
            if params is None:  # unsupported entity
                continue
            if params.size == 4:
                if np.isclose(params[:2], params[2:]).all():
                    # Skip zero-length line segment
                    continue
            ent2 = i2c_utils.entity_from_params(params)
            params2 = i2c_utils.parameterize_entity(ent2)
            assert_almost_equal(params, params2)


def test_quantize_params(sketches):
    n_bins = np.random.choice([64, 128, 256])
    all_bins = []
    for sketch in sketches:
        i2c_utils.normalize_sketch(sketch)
        for ent in sketch.entities.values():
            params = i2c_utils.parameterize_entity(ent)
            if params is None:
                continue
            bins = i2c_utils.quantize_params(params, n_bins)
            all_bins.append(bins)
    unique_bins = np.unique(np.concatenate(all_bins))
    assert np.array_equal(unique_bins, range(n_bins))


def test_dequantize_params(sketches):
    n_bins = np.random.choice([64, 128, 256])
    tol = (i2c_utils.MAX_VAL - i2c_utils.MIN_VAL) / n_bins / 2
    for sketch in sketches:
        i2c_utils.normalize_sketch(sketch)
        for ent in sketch.entities.values():
            params = i2c_utils.parameterize_entity(ent)
            if params is None:
                continue
            bins = i2c_utils.quantize_params(params, n_bins)
            params2 = i2c_utils.dequantize_params(bins, n_bins)
            diff = np.abs(params.round(decimals=10) - params2)
            assert (diff <= tol).all()


def test_noisify_sketch_ents(sketches):
    max_diff = 0.1
    for sketch in sketches:
        i2c_utils.normalize_sketch(sketch)
        all_ori_params = [i2c_utils.parameterize_entity(ent) for ent in
            sketch.entities.values()]
        if any(this_params is None for this_params in all_ori_params):
            continue
        noise_models.noisify_sketch_ents(sketch, std=0.2, max_diff=max_diff)
        all_new_params = [i2c_utils.parameterize_entity(ent) for ent in
            sketch.entities.values()]
        for ori_params, new_params in zip(all_ori_params, all_new_params):
            assert (new_params != ori_params).all()
            assert (np.abs(new_params - ori_params) <= max_diff).all()
