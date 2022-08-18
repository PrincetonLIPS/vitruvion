"""Filters sequences according to given criteria.

Usage:

.. code-block:: bash

    python -m img2cad.pipeline.filter_sequences sequence_file=/home/aseff/gencad/data/sequence_data/unique/sg_t16_train.npy

"""

import dataclasses
import io
import functools
import multiprocessing
import os
from typing import Optional

import numpy as np
import hydra
import omegaconf
import tqdm

from img2cad import data_utils
from sketchgraphs import data as datalib
from sketchgraphs.data import flat_array


@dataclasses.dataclass
class FilterConfiguration:
    min_entities: int = 6
    max_entities: int = 16


@dataclasses.dataclass
class FilterProcessConfiguration:
    sequence_file: str = omegaconf.MISSING
    output_path: str = 'filtered.npy'
    filter: FilterConfiguration = dataclasses.field(default_factory=FilterConfiguration)
    num_workers: Optional[int] = None


def _filter_sketch_render_valid(sketch: datalib.Sketch) -> bool:
    """Evaluates whether the sketch is valid from a rendering perspective.
    """
    try:
        scale = data_utils.normalize_sketch(sketch)
    except Exception:
        # Also discard sequences where normalize_sketch fails.
        return False

    if scale == -1:
        return False

    # Discard some pathological sketches which have invalid entities
    for ent in sketch.entities.values():
        # Discard zero-radius circles/arcs
        if hasattr(ent, 'radius'):
            if getattr(ent, 'radius') == 0:
                return False
        # Discard zero-length lines/arcs
        if isinstance(ent, datalib.Line):
            if np.allclose(ent.start_point, ent.end_point):
                return False
        if isinstance(ent, datalib.Arc):
            if np.allclose(ent.start_point, ent.mid_point):
                return False

    return True


def filter_sequence(sequence, config: FilterConfiguration) -> bool:
    """Returns whether the current sequence represents a valid sketch according to the config,
    and whether it can be normalized.
    """
    sketch = datalib.sketch_from_sequence(sequence)
    if len(sketch.entities) > config.max_entities or len(sketch.entities) < config.min_entities:
        return False

    return _filter_sketch_render_valid(sketch)


def _filter_instance_raw(sequence_bytes, config: FilterConfiguration) -> bool:
    sequence = flat_array.FlatSerializedArray.decode_raw_bytes(sequence_bytes)
    return filter_sequence(sequence, config)


@hydra.main(config_name='conf')
def main(config: FilterProcessConfiguration):
    sequence_path = hydra.utils.to_absolute_path(config.sequence_file)
    sequence_info = flat_array.load_dictionary_flat(sequence_path)
    sequences: flat_array.FlatSerializedArray = sequence_info['sequences']

    sequence_raw_bytes = (sequences.get_raw_bytes(i) for i in range(len(sequences)))

    filter_fn = functools.partial(_filter_instance_raw, config=config.filter)

    if config.num_workers == 0:
        pool = None
        result = map(filter_fn, sequence_raw_bytes)
    else:
        pool = multiprocessing.Pool(config.num_workers)
        result = pool.imap(filter_fn, sequence_raw_bytes, chunksize=128)

    result = tqdm.tqdm(result, total=len(sequences), smoothing=0.01)
    result = list(result)
    mask = np.array(result, dtype=np.bool_)

    print('Done filtering sequences, kept {}/{} ({:.2%})'.format(np.sum(mask), len(mask), np.mean(mask)))

    current_offset = 0
    offsets = [0]
    sequence_data = io.BytesIO()

    for i, r in enumerate(result):
        if not r:
            continue

        current_sequence_raw_data = sequences.get_raw_bytes(i)
        sequence_data.write(current_sequence_raw_data)
        current_offset += len(current_sequence_raw_data)
        offsets.append(current_offset)

    offsets = np.array(offsets, dtype=np.int64)

    new_sequences = flat_array.pack_list_flat(offsets, np.asarray(sequence_data.getbuffer()))

    result_dict = {
        'sequences': new_sequences,
        'sequence_lengths': sequence_info['sequence_lengths'][mask]
    }

    if 'counts' in sequence_info:
        result_dict['counts'] = sequence_info['counts'][mask]

    result = flat_array.pack_dictionary_flat(result_dict)

    output_path = os.path.abspath(config.output_path)

    print('Saving output in path {}'.format(output_path))
    np.save(output_path, result)


if __name__ == '__main__':
    from hydra.core.config_store import ConfigStore
    cs = ConfigStore()
    cs.store('conf', node=FilterProcessConfiguration)
    main()
