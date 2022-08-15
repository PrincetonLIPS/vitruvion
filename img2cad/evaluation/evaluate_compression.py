"""Draft script for compression evaluation
"""

import lzma

import numpy as np


def main():
    tokens = np.load('/mnt/ceph/users/wzhou/projects/gencad/img2cad/data/sg_filtered_unique_b64.cache.npz')
    compressed = lzma.compress(tokens['val'].tobytes())

    num_sketches = len(tokens['offsets']) - 1
    num_primitives = np.sum(tokens['pos'][tokens['offsets'][1:] - 1] - 1)

    bits_per_sketch = len(compressed) * 8 / num_sketches
    bits_per_primitive = len(compressed) * 8 / num_primitives

    print(f'Bits per sketch: {bits_per_sketch}. Bits per primitive {bits_per_primitive}.')
