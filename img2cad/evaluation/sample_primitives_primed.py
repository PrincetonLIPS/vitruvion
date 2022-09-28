"""Script for sampling primitive completions.
"""

import dataclasses
from typing import List, Dict, Optional

import matplotlib
import matplotlib.figure
import matplotlib.pyplot as plt
import hydra
import omegaconf
import tqdm
import torch

from img2cad import constraint_data, dataset, primitives_data, primitives_models, sample_img2prim
from img2cad.pipeline.prerender_images import render_sketch
from sketchgraphs import data as datalib

ENTITY_TYPES = (datalib.EntityType.Point, datalib.EntityType.Line, datalib.EntityType.Circle, datalib.EntityType.Arc)


def get_node_indices(seq) -> List[int]:
    return [i for i, op in enumerate(seq) if isinstance(op, datalib.NodeOp) and op.label in ENTITY_TYPES]

def obtain_sketch_prefix(sketch: datalib.Sketch, prefix_fraction: float=0.5) -> datalib.Sketch:
    """Create a sketch containing the given fraction of the entities in the provided sketch."""
    seq = datalib.sketch_to_sequence(sketch)
    node_indices = get_node_indices(seq)
    num_nodes_prefix = int(len(node_indices) * prefix_fraction)
    half_sketch = datalib.sketch_from_sequence(seq[:node_indices[num_nodes_prefix]])
    return half_sketch

def complete_sketch(model: primitives_models.RawPrimitiveModule, sketch: datalib.Sketch) -> datalib.Sketch:
    """Completes the given sketch using the provided model.
    """
    num_position_bins = model.hparams.data.num_position_bins

    tok_input, _ = dataset.tokenize_sketch(sketch, num_position_bins, include_stop=False)
    tok_completed = sample_img2prim.sample(model.model, 130, tok_input=tok_input)

    completed_sketch = dataset.sketch_from_tokens(tok_completed, num_position_bins)
    return completed_sketch


def process_single_sketch(model: primitives_models.RawPrimitiveModule, sketch: datalib.Sketch, prefix_fraction: float, num_samples_per: int=1):
    sketch_prefix = obtain_sketch_prefix(sketch, prefix_fraction)
    sketch_completions = [
        complete_sketch(model, sketch_prefix) for _ in range(num_samples_per)]

    return {
        'original': sketch,
        'prefix': sketch_prefix,
        'completed': sketch_completions
    }


def _plot_sketches(sketches: Dict[str, datalib.Sketch], prefix: str=None):
    if prefix is None:
        prefix = ''

    for k, sketch in sketches.items():
        if k == 'completed':
            # sketch is a list in this case
            for samp_idx, completed_sketch in enumerate(sketch):
                fig = render_sketch(completed_sketch, return_fig=True,
                    sketch_extent=1)
                filename = '%s_%s_%02i.pdf' % (prefix, k, samp_idx)
                fig.savefig(filename, dpi=128)
        else:
            fig = render_sketch(sketch, return_fig=True, sketch_extent=1)
            fig.savefig(f'{prefix}_{k}.pdf', dpi=128)


@dataclasses.dataclass
class SamplePrimitivesPrimedConfig:
    checkpoint_path: str = omegaconf.MISSING
    prefix_fraction: float = 0.5
    limit_sketches: int = 100
    num_samples_per: int = 1
    sequence_path: Optional[str] = None


@hydra.main(config_name='conf', config_path=None)
def main(config: SamplePrimitivesPrimedConfig):
    model: primitives_models.RawPrimitiveModule = primitives_models.RawPrimitiveModule.load_from_checkpoint(
        hydra.utils.to_absolute_path(config.checkpoint_path))

    model = model.eval()
    model = model.to('cuda', dtype=torch.float16)

    if config.sequence_path is not None:
        # Override hparams stored sequence path
        sequence_path = hydra.utils.to_absolute_path(config.sequence_path)
    else:
        sequence_path = model.hparams.data.sequence_path

    dataset = constraint_data.ConstraintDataset(
        sequence_path,
        model.hparams.data.num_position_bins,
        model.hparams.data.max_token_length,
        tokenize=False)

    _, _, dataset = primitives_data.split_dataset(
        dataset, model.hparams.data.validation_fraction, model.hparams.data.test_fraction)

    num_digits = len(f'{config.limit_sketches}')
    format_specifier = f'{{:0{num_digits}d}}'

    for i in tqdm.trange(min(len(dataset), config.limit_sketches)):
        sketches = process_single_sketch(model, dataset[i], config.prefix_fraction, config.num_samples_per)
        _plot_sketches(sketches, format_specifier.format(i))
        plt.close('all')


if __name__ == '__main__':
    from hydra.core.config_store import ConfigStore
    cs = ConfigStore()
    cs.store(name='conf', node=SamplePrimitivesPrimedConfig)
    main()
