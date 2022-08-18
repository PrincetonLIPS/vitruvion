"""Computes unique token sequences.
"""

import numpy as np


def compute_unique_indexes(data: np.ndarray, offsets: np.ndarray) -> np.ndarray:
    """Computes indices of unique sequences in the data.

    We represent the data as a combination of a flat array, as well as offsets into
    that array representing each sequence. Such data is for example created by the caching
    mechanism of the primitive token dataset.

    Parameters
    ----------
    data : np.ndarray
        1-dimensional array of tokens, representing concatenate sequences to uniqueify.
    offsets : np.ndarray
        Offsets into the ``data`` array representing sequence boundaries.

    Returns
    -------
    np.ndarray
        A 1-dimensional integer array representing the indices of unique sequences in the provided data.
    """

    lengths = np.diff(offsets)

    unique_lengths = np.unique(lengths)

    unique_idxs = []

    for length in unique_lengths:
        idx = np.flatnonzero(lengths == length)
        idx_expand = np.expand_dims(offsets[idx], -1) + np.arange(length)

        sequences = data[idx_expand]
        _, idx_unique = np.unique(sequences, axis=0, return_index=True)
        unique_idxs.append(idx[idx_unique])

    unique_idxs = np.concatenate(unique_idxs)
    unique_idxs.sort()

    return unique_idxs
