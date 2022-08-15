"""This script is responsible for creating primitive sequence tokens, and extracting unique sequences
among the computed sequence tokens.

"""

import dataclasses
import logging
import os

from typing import Dict, Sequence

import hydra
import numpy as np
import omegaconf

from hydra.core.config_store import ConfigStore

from img2cad import dataset
from sketchgraphs.data import flat_array


@dataclasses.dataclass
class TokenizeSequenceConfig:
    sequence_file: str = omegaconf.MISSING
    num_position_bins: int = 64


def _offsets_from_counts(counts: np.ndarray):
    offsets = np.zeros(len(counts) + 1, dtype=np.int64)
    np.cumsum(counts, out=offsets[1:])
    return offsets


def _get_cache_data_path(sequence_path: os.PathLike, num_position_bins: int) -> str:
    path, _ = os.path.splitext(sequence_path)
    return path + f'_unique_b{num_position_bins}.cache.npz'


def _get_unique_sequence_path(sequence_path: os.PathLike, num_position_bins: int) -> str:
    path, _ = os.path.splitext(sequence_path)
    return path + f'_unique.npy'


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


@hydra.main(config_name='conf')
def main(config: TokenizeSequenceConfig):
    logger = logging.getLogger(__name__)
    primitive_dataset = dataset.PrimitiveDataset(config.sequence_file, config.num_position_bins)

    data = dataset.process_sequence_data(primitive_dataset)
    data = {k: np.asarray(v) for k, v in data.items()}

    counts = np.diff(data['offsets'])

    logger.info('Uniqueifying token sequences.')
    unique_idx = compute_unique_indexes(data['val'], data['offsets'])
    logger.info(f'Kept {len(unique_idx)} / {len(counts)} sequences ({len(unique_idx) / len(counts):.2%}).')

    logger.info('Subsampling computed token sequences.')
    data = subsample_tokens(data, unique_idx)
    cache_data_path = _get_cache_data_path(config.sequence_file, config.num_position_bins)
    logger.info(f'Saving unique token sequences at {cache_data_path}')
    np.savez(cache_data_path, **data)

    logger.info('Subsampling sequences.')
    sequence_output_path = _get_unique_sequence_path(config.sequence_file, config.num_position_bins)
    sequence_data = flat_array.load_dictionary_flat(config.sequence_file)
    sequence_data = subsample_sequences(sequence_data, unique_idx)
    logger.info(f'Saving unique sequences at {sequence_output_path}')
    np.save(sequence_output_path, sequence_data, allow_pickle=False)


def subsample_sequences(sequence_data, idx: Sequence[int]) -> np.ndarray:
    sketch_ids = sequence_data['sketch_ids'][idx]
    sequence_lengths = sequence_data['sequence_lengths'][idx]

    sequences: flat_array.FlatSerializedArray = sequence_data['sequences']
    idx_sequences_data = [sequences.get_raw_bytes(i) for i in idx]
    counts = [len(x) for x in idx_sequences_data]

    sequences = flat_array.pack_list_flat(_offsets_from_counts(counts), np.concatenate(idx_sequences_data))

    return flat_array.pack_dictionary_flat({
        'sequences': sequences,
        'sketch_ids': sketch_ids,
        'sequence_lengths': sequence_lengths
    })


def subsample_tokens(tokens_data: Dict[str, np.ndarray], idx: Sequence[int]) -> Dict[str, np.ndarray]:
    offsets = tokens_data['offsets']
    counts = np.zeros(len(idx))

    result = {
        **tokens_data
    }

    for k in ('val', 'pos', 'coord'):
        data = tokens_data[k]
        result_k = []

        for j, i in enumerate(idx):
            o_start = offsets[i]
            o_end = offsets[i + 1]
            counts[j] = o_end - o_start
            result_k.append(data[o_start:o_end])

        result[k] = np.concatenate(result_k)

    result['offsets'] = _offsets_from_counts(counts)
    return result


if __name__ == '__main__':
    cs = ConfigStore()
    cs.store('conf', node=TokenizeSequenceConfig)
    main()

