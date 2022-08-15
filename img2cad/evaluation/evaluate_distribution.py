"""Script for evaluating distribution of primitives and constraints.
"""

import dataclasses
import glob
import logging
import pickle

from typing import List, Union, Sequence


import hydra
import numpy as np
import matplotlib.pyplot as plt
import omegaconf

import img2cad.constraint_data
from sketchgraphs import data as datalib
from sketchgraphs.data import flat_array
from sketchgraphs.data.dof import get_sequence_dof

CB_COLORS = ['#377eb8', '#ff7f00', '#4daf4a', '#f781bf', '#a65628', '#984ea3', '#999999', '#e41a1c', '#dede00']
FONTSIZE = 36


log = logging.getLogger(__name__)


def is_entity_op(op):
    return (isinstance(op, datalib.NodeOp) and not isinstance(op.label, datalib.SubnodeType) and op.label not in (datalib.EntityType.Stop, datalib.EntityType.External))
  
def is_constraint_op(op):
    return (isinstance(op, datalib.EdgeOp) and op.label != datalib.ConstraintType.Subnode)

def count_primitives(seq):
    return sum(1 for op in seq if is_entity_op(op))

def count_constraints(seq):
    return sum(1 for op in seq if is_constraint_op(op))

def get_dof_stats(seq):
    dof_seq = get_sequence_dof(seq)
    node_dof = np.maximum(0, dof_seq).sum(0)
    edge_dof = -np.minimum(0, dof_seq).sum(0)
    return node_dof, edge_dof

def compute_sample_stats(seqs):
    out = {}
    out['num_entities'] = np.array([count_primitives(seq) for seq in seqs])
    out['num_constraints'] = np.array([count_constraints(seq) for seq in seqs])
    node_dof, edge_dof = zip(*[get_dof_stats(seq) for seq in seqs])
    out['entity_dof'] = np.array(node_dof)
    out['constraint_dof'] = np.array(edge_dof)

    out['net_dof'] = out['entity_dof'] - out['constraint_dof']

    return out

def resample(vals):
    vals = np.asarray(vals)
    indices = np.random.randint(0, len(vals), len(vals))
    return vals[indices]

def get_int_frequencies(x, n):
    return np.bincount(x, minlength=n)[:n] / len(x)


def compare_counts(gt_counts, sample_counts, nbins: int, ax=None):
    bins = np.arange(nbins)
    gt_freqs = get_int_frequencies(gt_counts, nbins)
    sample_freqs = get_int_frequencies(sample_counts, nbins)
    sample_freqs_dist = np.stack(
        [get_int_frequencies(resample(sample_counts), nbins)
        for _ in range(2000)], axis=-1)
    sample_freqs_lo = np.percentile(sample_freqs_dist, 5, axis=-1)
    sample_freqs_hi = np.percentile(sample_freqs_dist, 95, axis=-1)

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))
    else:
        fig = None

    ax.set_xlim(0, nbins-1)
    ax.plot(bins, sample_freqs, drawstyle='steps', color=CB_COLORS[1])
    ax.plot(bins, sample_freqs_lo, drawstyle='steps', color=CB_COLORS[1], label='samples')
    ax.plot(bins, sample_freqs_hi, drawstyle='steps', color=CB_COLORS[1])
    ax.fill_between(bins, sample_freqs_lo, sample_freqs_hi, color=CB_COLORS[1], step='pre', alpha=.3)
    ax.plot(bins, gt_freqs, drawstyle='steps', color=CB_COLORS[0], label='dataset')

    ax.set_ylabel('Frequency')
    ax.legend()

    return fig, ax

def load_generated_sketches(path: str) -> List[datalib.Sketch]:
    files = glob.glob(path + '/**/constraints.pkl')

    result = []

    for filepath in files:
        with open(filepath, 'rb') as f:
            result.extend(pickle.load(f))

    result = [r[0] for r in result if r is not None]
    return result


_CATEGORICAL_CONSTRAINTS = set(
    datalib.ConstraintType[v.name]
    for v in img2cad.constraint_data.Token
    if v not in (
        img2cad.constraint_data.Token.Pad,
        img2cad.constraint_data.Token.Start,
        img2cad.constraint_data.Token.Stop))

def filter_sequence_constraints(seqs: Sequence[Sequence[Union[datalib.EdgeOp, datalib.NodeOp]]]):
    """Filter sequences to remove all constraints which are not modelled by the img2cad system."""
    def _filter_not_numerical_constraint(op: Union[datalib.NodeOp, datalib.EdgeOp]):
        return isinstance(op, datalib.NodeOp) or op.label in _CATEGORICAL_CONSTRAINTS

    def _filter_sequence_single(seq: Sequence[Union[datalib.NodeOp, datalib.EdgeOp]]):
        return [op for op in seq if _filter_not_numerical_constraint(op)]

    return [_filter_sequence_single(s) for s in seqs]


def plot_and_save(name: str, true_counts: Sequence[int], sample_counts: Sequence[int], nbins: int, xlims=None, xlabel=None):
    fig, ax = compare_counts(true_counts, sample_counts, nbins)

    if xlims is not None:
        ax.set_xlim(*xlims)

    if xlabel is not None:
        ax.set_xlabel(xlabel)

    fig.tight_layout()
    fig.savefig(f'{name}.png')
    fig.savefig(f'{name}.pdf')


@dataclasses.dataclass
class EvaluateDistributionConfig:
    sequence_path: str = omegaconf.MISSING
    generated_path: str = omegaconf.MISSING


@hydra.main(config_name='conf')
def main(config: EvaluateDistributionConfig):
    rng = np.random.Generator(np.random.PCG64(42))

    log.info('Loading reference sequences.')
    sequences = flat_array.load_dictionary_flat(hydra.utils.to_absolute_path(config.sequence_path))['sequences']
    subsample = [sequences[i] for i in rng.choice(len(sequences), replace=False, size=10000)]
    subsample = filter_sequence_constraints(subsample)

    log.info('Computing counts for reference sequences.')
    counts_truth = compute_sample_stats(subsample)

    log.info('Loading generated sketches')
    generated_sketches = load_generated_sketches(hydra.utils.to_absolute_path(config.generated_path))
    log.info('Converting generated sketches to sequences')
    generated_sequences = list(map(datalib.sketch_to_sequence, generated_sketches))
    counts = compute_sample_stats(generated_sequences)

    plot_and_save('entities', counts_truth['num_entities'], counts['num_entities'], 16, xlims=(6, 16), xlabel='Number of primitives')
    plot_and_save('constraints', counts_truth['num_constraints'], counts['num_constraints'], 50, xlabel='Number of constraints')
    plot_and_save('entity_dof', counts_truth['entity_dof'], counts['entity_dof'], 80, xlabel='Total DOF')
    plot_and_save('constraint_dof', counts_truth['constraint_dof'], counts['constraint_dof'], 80, xlabel='DOF removed by constraints')
    plot_and_save('net_dof', np.maximum(counts_truth['net_dof'], 0), np.maximum(counts['net_dof'], 0), 30, xlabel='Net DOF')


if __name__ == '__main__':
    from hydra.core.config_store import ConfigStore
    cs = ConfigStore()
    cs.store(name='conf', node=EvaluateDistributionConfig)
    main()

