"""This module handles constraint dataset functionality including tokenization.
"""

import copy
import dataclasses
import enum
from typing import Dict, Optional, Sequence

import numpy as np
import pytorch_lightning
import torch
import torch.utils.data

import sketchgraphs.data as datalib
from sketchgraphs.data import flat_array

from img2cad import noise_models, data_utils, dataset, modules, primitives_data
from img2cad.dataset import _pad_or_truncate_to_length, NON_COORD_TOKEN


# Constraint coord tokens indicate parameter type (only reference params atm)
CONSTRAINT_COORD_TOKENS = [NON_COORD_TOKEN+1, NON_COORD_TOKEN+2]  # [2, 3]

@dataclasses.dataclass
class PrimitiveNoiseConfig:
    """Configuration for primitive noise model.
    The primitive noise is implemented as a truncated normal noise.

    Attributes
    ----------
    enabled : bool
        Whether to use primitive noise model
    std : float
        Standard deviation of noise to add
    max_difference : float
        Maximum difference between noisy coordinate and original coordinate.
    """
    enabled: bool = True
    std: float = 0.15
    max_difference: float = 0.15


class Token(enum.IntEnum):
    """Enumeration indicating the non-parameter value tokens of ConstraintModel.

    At the moment, only categorical constraints are considered.
    """
    Pad = 0
    Start = 1
    Stop = 2
    Coincident = 3
    Concentric = 4
    Equal = 5
    Fix = 6
    Horizontal = 7
    Midpoint = 8
    Normal = 9
    Offset = 10
    Parallel = 11
    Perpendicular = 12
    Quadrant = 13
    Tangent = 14
    Vertical = 15



def tokenize_constraints(seq: datalib.ConstructionSequence, gather_idxs: Sequence[int], max_length: Optional[int]=None):
    """Tokenizes the constraints in a sketch construction sequence.

    Parameters
    ----------
    seq : datalib.ConstructionSequence
        The sketch construction sequence to tokenize.
    gather_idxs : Sequence[int]
        Indices produced by `dataset.tokenize_sketch` used to track entity tokens.
    max_length : Optional[int]
        If not `None`, truncates the sequence to this length.
    """
    val_tokens = [Token.Start]
    coord_tokens = [NON_COORD_TOKEN]
    pos_idx = 1  # 0 is reserved for padding
    pos_tokens = [pos_idx]

    # Iterate through edge ops
    for op in seq:
        # Ensure op is applicable edge op
        if not isinstance(op, datalib.EdgeOp):
            continue
        if not op.label.name in Token.__members__:
            continue
        refs = op.references
        if 0 in refs:  # skip external constraints
            continue

        # Add constraint type tokens
        val_tokens.append(Token[op.label.name])
        coord_tokens.append(NON_COORD_TOKEN)
        pos_idx += 1
        pos_tokens.append(pos_idx)

        # Add reference parameters
        val_tokens.extend(
            [gather_idxs[ref] + len(Token) for ref in sorted(refs)])
        coord_tokens.extend(CONSTRAINT_COORD_TOKENS[:len(refs)])
        pos_tokens.extend([pos_idx] * len(refs))
    val_tokens.append(Token.Stop)
    coord_tokens.append(NON_COORD_TOKEN)
    pos_tokens.append(pos_idx+1)

    sample = {
        'val': _pad_or_truncate_to_length(np.array(val_tokens, dtype=np.int64), max_length),
        'coord': _pad_or_truncate_to_length(np.array(coord_tokens, dtype=np.int64), max_length),
        'pos': _pad_or_truncate_to_length(np.array(pos_tokens, dtype=np.int64), max_length)
    }

    return sample


def apply_primitive_noise(sketch: datalib.Sketch, std: float=0.15, max_difference: float=0.15) -> datalib.Sketch:
    noise_sketch = copy.deepcopy(sketch)
    try:
        noise_models.noisify_sketch_ents(noise_sketch, std=std, max_diff=max_difference)
    except:
        noise_sketch = sketch
    return noise_sketch


class ConstraintDataset(torch.utils.data.Dataset[Dict[str, torch.Tensor]]):
    """Constraint generation dataset."""

    def __init__(self, sequence_file: str, num_bins: int, max_length: Optional[int]=None,
                 primitive_noise_config: PrimitiveNoiseConfig=None, tokenize: bool=True):
        """Create a new constraint dataset.

        Parameters
        ----------
        sequence_file : str
            Path to the sequence file to load.
        num_bins : int
            Number of bins for positional quantization.
        max_length : int, optional
            Length to which to pad or truncate sequence tokens.
        primitive_noise_config : PrimitiveNoiseConfig, optional
            If not `None`, configuration for the primitive noise to apply.
        tokenize : bool
            If `True` (default), indicates that the sketches should be tokenized into
            primitive and constraint token sequences. Otherwise, simply returns the sketch
            (with potential primitive noise applied).
        """

        if primitive_noise_config is None:
            primitive_noise_config = PrimitiveNoiseConfig(enabled=False)

        self.primitive_noise_config = primitive_noise_config
        self.num_bins = num_bins
        self.max_length = max_length

        self.sequences = flat_array.load_dictionary_flat(sequence_file)['sequences']
        self.tokenize = tokenize

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        seq = self.sequences[idx]

        sketch = datalib.sketch_from_sequence(seq)
        data_utils.normalize_sketch(sketch)

        if self.primitive_noise_config.enabled:
            sketch = apply_primitive_noise(sketch, self.primitive_noise_config.std, self.primitive_noise_config.max_difference)

        if not self.tokenize:
            return sketch

        sample, gather_idx = dataset.tokenize_sketch(sketch, self.num_bins, self.max_length)
        c_sample = tokenize_constraints(seq, gather_idx, self.max_length)
        c_sample = {f'c_{k}': v for k, v in c_sample.items()}

        sample = {
            **sample,
            **c_sample
        }

        sample = {k: torch.from_numpy(v) for k, v in sample.items()}
        return sample

    def __len__(self) -> int:
        return len(self.sequences)


@dataclasses.dataclass
class ConstraintDataConfig(primitives_data.PrimitiveDataConfig):
    max_token_length: int = 130
    primitive_noise: PrimitiveNoiseConfig = PrimitiveNoiseConfig(enabled=True)


class ConstraintDataModule(pytorch_lightning.LightningDataModule):
    def __init__(self, config: ConstraintDataConfig, batch_size: int, num_workers: int=8):
        super().__init__()

        self.batch_size = batch_size
        self.config = config
        self.num_workers = num_workers

        self._dataset = None
        self._train_dataset = None
        self._valid_dataset = None
        self._test_dataset = None

    def setup(self, stage: Optional[str]=None) -> None:
        self._dataset = ConstraintDataset(
            self.config.sequence_path,
            self.config.num_position_bins,
            self.config.max_token_length,
            self.config.primitive_noise)

        self._train_dataset, self._valid_dataset, self._test_dataset = primitives_data.split_dataset(
            self._dataset, self.config.validation_fraction, self.config.test_fraction)

    def _make_dataloader(self, dataset, shuffle: bool) -> torch.utils.data.DataLoader[modules.TokenInput]:
        if dataset is None:
            return None

        return torch.utils.data.DataLoader(
            dataset, self.batch_size,
            shuffle=shuffle, num_workers=self.num_workers,
            pin_memory=False,
            persistent_workers=self.num_workers > 0)

    def train_dataloader(self) -> torch.utils.data.DataLoader[modules.TokenInput]:
        return self._make_dataloader(self._train_dataset, shuffle=True)

    def val_dataloader(self) -> Optional[torch.utils.data.DataLoader[modules.TokenInput]]:
        return self._make_dataloader(self._valid_dataset, shuffle=False)

    def test_dataloader(self) -> Optional[torch.utils.data.DataLoader[modules.TokenInput]]:
        return self._make_dataloader(self._test_dataset, shuffle=False)

    @property
    def train_dataset_size(self) -> int:
        return len(self._train_dataset)

