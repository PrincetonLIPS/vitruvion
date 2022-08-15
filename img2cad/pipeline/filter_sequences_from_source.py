"""Filter sequences from source shards.

The script is responsible from creating the base sequence file from the json shards.
This is the first step of our processing pipeline, simply selecting the sketches to train on.
"""

import dataclasses
import enum
import functools
import os

import numpy as np
import hydra
import omegaconf

from hydra.core.config_store import ConfigStore

from sketchgraphs import data as datalib
from sketchgraphs.pipeline import make_sequence_dataset
from sketchgraphs.pipeline.make_sequence_dataset import FilterReason

from img2cad import data_utils


@dataclasses.dataclass
class FilterConfiguration:
    min_entities: int = 6
    max_entities: int = 16

@dataclasses.dataclass
class FilterSequenceRunConfiguration:
    filter: FilterConfiguration = FilterConfiguration()
    input_folder: str = omegaconf.MISSING
    output_folder: str = omegaconf.MISSING
    total_sketches: int = 16261381


class ExtendedFilterReason(enum.Enum):
    CannotNormalize = 8
    ZeroSizedEntity = 9


def _filter_sketch_render_valid(sketch: datalib.Sketch) -> FilterReason:
    """Evaluates whether the sketch is valid from a rendering perspective.
    """
    try:
        scale = data_utils.normalize_sketch(sketch)
    except Exception:
        # Also discard sequences where normalize_sketch fails.
        return ExtendedFilterReason.CannotNormalize

    if scale == -1:
        return ExtendedFilterReason.CannotNormalize

    # Discard some pathological sketches which have zero-sized entities
    for ent in sketch.entities.values():
        if hasattr(ent, 'radius'):
            if getattr(ent, 'radius') == 0:
                return ExtendedFilterReason.ZeroSizedEntity
        # Discard zero-length lines/arcs
        if isinstance(ent, datalib.Line):
            if np.allclose(ent.start_point, ent.end_point):
                return ExtendedFilterReason.ZeroSizedEntity
        if isinstance(ent, datalib.Arc):
            if np.allclose(ent.start_point, ent.mid_point):
                return ExtendedFilterReason.ZeroSizedEntity

    return FilterReason.Accepted


def filter_sketch(sketch: datalib.Sketch, config: FilterConfiguration) -> FilterReason:
    config_s = make_sequence_dataset.make_default_filter_config(config.min_entities, config.max_entities)
    reason = make_sequence_dataset.filter_sketch(sketch, config_s)

    if reason != FilterReason.Accepted:
        return reason

    return _filter_sketch_render_valid(sketch)


def _list_directory(directory: str):
    files = os.listdir(directory)
    return [os.path.join(directory, f) for f in files]


@hydra.main(config_name='conf')
def main(config: FilterSequenceRunConfiguration):
    shards = _list_directory(hydra.utils.to_absolute_path(config.input_folder))
    filter_function = functools.partial(filter_sketch, config=config.filter)

    num_threads = len(os.sched_getaffinity(0))

    result = make_sequence_dataset.process(shards, num_threads, filter_function, total_sketches=config.total_sketches)

    output_path = os.path.join(hydra.utils.to_absolute_path(config.output_folder), 'sg_filtered.npy')
    print('Saving result at {0}'.format(output_path))
    np.save(output_path, result, allow_pickle=False)


if __name__ == '__main__':
    cs = ConfigStore()
    cs.store('conf', node=FilterSequenceRunConfiguration)
    main()
