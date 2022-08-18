"""This module handles dataset-related functionality including tokenization.
"""

import copy
import enum
import logging
import io
import os
import sys
from random import shuffle
from collections import OrderedDict
from typing import Callable, Dict, List, Union, Sequence, Tuple, Optional

if sys.version_info >= (3, 8):
    from typing import TypedDict
else:
    from typing_extensions import TypedDict

import numpy as np
import PIL
import PIL.Image

import torch
import torch.nn.functional
import torch.utils.data
import tqdm

import sketchgraphs.data as datalib
from sketchgraphs.data import flat_array
from sketchgraphs.data import Arc, Circle, Line, Point
from . import data_utils as i2c_utils
from .data_utils import NUM_PARAMS
from . import noise_models


INCLUDE_CONSTRUCTION = True


NON_COORD_TOKEN = 1  # 0 is reserved for padding
COORD_TOKEN_MAP = {}
tok = NON_COORD_TOKEN + 1
for ent_type in [Arc, Circle, Line, Point]:
    COORD_TOKEN_MAP[ent_type] = list(range(tok, tok+NUM_PARAMS[ent_type]))
    tok += NUM_PARAMS[ent_type]


class Token(enum.IntEnum):
    """Enumeration indicating the non-parameter value tokens of PrimitiveModel.
    """
    Pad = 0
    Start = 1
    Stop = 2
    Arc = 3
    Circle = 4
    Line = 5
    Point = 6


def _pad_or_truncate_to_length(arr: np.ndarray, target_length: Optional[int]=None) -> np.ndarray:
    """Ensures that the array has the given length by truncating or padding.

    If `target_length` is `None`, returns `arr` unchanged.
    """

    if target_length is None:
        return arr

    if len(arr) > target_length:
        return arr[:target_length]

    if isinstance(arr, np.ndarray):
        return np.pad(arr, (0, target_length - len(arr)), constant_values=Token.Pad)
    elif isinstance(arr, torch.Tensor):
        return torch.nn.functional.pad(arr, (0, target_length - len(arr)), value=Token.Pad)
    else:
        raise ValueError('arr must be either numpy array or torch Tensor')


class SketchTokenSet(TypedDict):
    val: np.ndarray
    coord: np.ndarray
    pos: np.ndarray


def tokenize_sketch(sketch: datalib.Sketch, num_bins: int, max_length: Optional[int]=None, permute: bool=False, include_stop: bool=True) -> Tuple[SketchTokenSet, List[int]]:
    """Tokenizes the given sketch.

    Parameters
    ----------
    sketch : datalib.Sketch
        Sketch from which to generate the tokens.
    num_bins : int
        Number of bins used to quantize sketch positions.
    max_length : int, optional
        Length to which the token sequence is padded or truncated.
    permute : bool, optional
        If True, the primitives are randomly permuted. (default False)
    include_stop : bool, optional
        If `True`, indicates that the stop token is included at the end of the
        sequence. Otherwise, no stop token is included at the end of the sequence.

    Returns
    -------
    data : SketchTokenSet
        A dictionary containing the token sequence corresponding to the given sketch.
        The token sequence is padded or truncated to the given length if specified.
    gather_idx : List[int]
        A list of integers representing indices to track for the constraint model.
    """
    val_tokens = [Token.Start]
    coord_tokens = [NON_COORD_TOKEN]
    pos_idx = 1  # 0 is reserved for padding
    pos_tokens = [pos_idx]

    # Index tracking for constraint model's gather operation
    gather_map = {
        Arc: [0, 1, 3, 5],
        Circle: [0, 1],
        Line: [0, 1, 3],
        Point: [0]
    }
    gather_idxs = [0]  # 0 is for external even though we don't use

    # isConstruction tokens
    construction_tok_dict = {
        True: len(Token) + num_bins,
        False: len(Token) + num_bins + 1
    }

    if permute:
        # Randomly permute primitive ordering
        entities = list(sketch.entities.items())
        shuffle(entities)
        sketch.entities = OrderedDict(entities)

    for ent in sketch.entities.values():
        gather_idxs.extend(
            [len(val_tokens) + gidx for gidx in gather_map[type(ent)]])
        val_tokens.append(Token[ent.type.name])
        coord_tokens.append(NON_COORD_TOKEN)
        pos_idx += 1
        pos_tokens.append(pos_idx)
        params = i2c_utils.parameterize_entity(ent)

        try:
            param_bins = i2c_utils.quantize_params(params, num_bins)
        except:
            # Zero-length line segment issue
            # if not self.noisify:
            #     raise ValueError("legal parameter value error")
            params[params > 0.5] = 0.5
            params[params < -0.5] = -0.5
            param_bins = i2c_utils.quantize_params(params, num_bins)

        val_tokens.extend(param_bins + len(Token))
        coord_tokens.extend(COORD_TOKEN_MAP[type(ent)])
        pos_tokens.extend([pos_idx] * param_bins.size)

        # Add isConstruction attribute
        if INCLUDE_CONSTRUCTION:
            val_tokens.append(construction_tok_dict[ent.isConstruction])
            coord_tokens.append(NON_COORD_TOKEN)
            pos_tokens.append(pos_idx)

    if include_stop:
        val_tokens.append(Token.Stop)
        coord_tokens.append(NON_COORD_TOKEN)
        pos_tokens.append(pos_idx+1)

    sample = {
        'val': _pad_or_truncate_to_length(np.array(val_tokens, dtype=np.int64), max_length), 
        'coord': _pad_or_truncate_to_length(np.array(coord_tokens, dtype=np.int64), max_length), 
        'pos': _pad_or_truncate_to_length(np.array(pos_tokens, dtype=np.int64), max_length)
    }

    return sample, gather_idxs


def param_seq_from_tokens(tokens, num_bins):
    """Convert value tokens to sequence of quantized entity parameters.

    Parameters
    ----------
    tokens : np.array
        The sequence of value tokens
    num_bins : int
        The number of quantization bins

    Returns
    -------
    all_params : list
        A list of tuples of the quantized entity parameters and isConstruction
        attribute, where all_params[i] contains the parameters for
        the i-th entity
    """

    reverse_construction_toks = {
        len(Token) + num_bins: True,
        len(Token) + num_bins + 1: False
    }

    all_params = []
    curr_params = []
    for token in tokens:
        if token == Token.Start:
            continue
        if token < len(Token):
            if curr_params:
                all_params.append((curr_params, isConstruction))
                curr_params = []
        if token in [Token.Stop, Token.Pad]:
            break
        if token >= len(Token):
            isConstruction = False  # initialize to False in case not modeling
            if token <= len(Token) + (num_bins-1):
                # Numerical coordinate
                curr_params.append(token - len(Token))
            else:
                # isConstruction attribute
                isConstruction = reverse_construction_toks[token]

    if curr_params:
        # Append possibly leftover entity parameters
        all_params.append((curr_params, isConstruction))
    return all_params


def sketch_from_tokens(tokens, num_bins):
    """Convert value tokens to Sketch instance.

    Parameters
    ----------
    tokens : np.array
        The sequence of value tokens
    num_bins : int
        The number of quantization bins

    Returns
    -------
    Sketch
        A Sketch instance corresponding to the given value tokens
    """
    all_params = param_seq_from_tokens(tokens, num_bins)
    sketch = datalib.Sketch()
    for idx, (ent_params, isConstruction) in enumerate(all_params):
        ent_params = i2c_utils.dequantize_params(ent_params, num_bins)
        ent_params = ent_params.tolist()
        ent = i2c_utils.entity_from_params(ent_params)
        if ent is not None:
            ent.entityId = str(idx+1)
            ent.isConstruction = isConstruction
        sketch.entities[str(idx)] = ent
    return sketch


def _counts_to_offset(x):
    x = np.asarray(x)
    offsets = np.zeros(len(x) + 1, dtype=np.int32)
    np.cumsum(x, out=offsets[1:])
    return offsets


def _adjust_indexes(indexes) -> List[np.ndarray]:
    result = []
    offset = 0
    for idx in indexes:
        result.append(idx + offset)
        offset += idx[-1]

    return result


class MultiImageDataset:
    def __init__(self, image_files: Union[str, Sequence[str]], adjust_shard_indexes: bool=False):
        """This class encapsulates a dataset of images assembled from multiple sharded files.

        Parameters
        ----------
        image_files : str or Sequence[str]
            Path to the shards containing image files.
        adjust_shard_indexes : bool
            If `True`, indexes into the shards are adjusted according to the order in which
            the shards are presented. Otherwise, indexes in each shard are left as is.
        """
        if isinstance(image_files, str):
            image_files = [image_files]

        self.image_data = [flat_array.load_dictionary_flat(f) for f in image_files]
        self._offsets = _counts_to_offset([len(d['imgs']) for d in self.image_data])

        indexes = [d['indexes'] for d in self.image_data]
        if adjust_shard_indexes:
            indexes = _adjust_indexes(indexes)
        self._all_indexes = np.concatenate(indexes)
        self._indexes = np.unique(self._all_indexes)

        if not np.all(np.diff(self._all_indexes) >= 0):
            raise ValueError('Image files not provided in sorted order!')

    def __getitem__(self, idx: int) -> Sequence[bytes]:
        segment_start, segment_end = np.searchsorted(self._all_indexes, [idx, idx + 1])
        array_idx = np.searchsorted(self._offsets, segment_start, side='right') - 1

        num_images = segment_end - segment_start

        idx_in_array = segment_start - self._offsets[array_idx]
        return self.image_data[array_idx]['imgs'][idx_in_array:idx_in_array + num_images]

    def __len__(self):
        return len(self._indexes)

    @property
    def indexes(self) -> np.ndarray:
        """Returns an integer array representing the indices of the images contained in the dataset.
        """
        return self._indexes


class PrimitiveDataset(torch.utils.data.Dataset[SketchTokenSet]):
    """This dataset encapsulates a sequence of sketch as a dataset of tokenized primitives."""
    def __init__(self, sequence_file: str, num_bins: int, max_length: Optional[int]=None, permute: bool=False):
        """Create a new primitive dataset from the given sequence.

        Parameters
        ----------
        sequence_file : str
            Path to the sequence file containing the sketch data.
        num_bins : int
            Number of bins to used for quantizing positional features.
        max_length : int, optional
            If not `None`, the length to which the sequences are padded or truncated.
            Otherwise, no padding / truncation is performed.
        permute : bool, optional
            If True, the primitives are randomly permuted.
            Note, this breaks constraint references. (default False)
        """
        super().__init__()

        self.sequences = flat_array.load_dictionary_flat(sequence_file)['sequences']
        self.max_length = max_length
        self.num_bins = num_bins
        self.permute = permute

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        seq = self.sequences[idx]

        sketch = datalib.sketch_from_sequence(seq)
        i2c_utils.normalize_sketch(sketch)

        sample, _ = tokenize_sketch(sketch, self.num_bins, self.max_length,
            permute=self.permute)
        sample = {k: torch.from_numpy(v) for k, v in sample.items()}
        return sample

    def __len__(self) -> int:
        return len(self.sequences)


def concatenate_tokens(tokens: Sequence[SketchTokenSet]) -> Dict[str, torch.Tensor]:
    """Concatenates a sequence of token sequences.

    Parameters
    ----------
    tokens : Sequence[SketchTokenSet]
        A sequence of tokens to concatenate.
    """
    counts = np.concatenate([
        np.diff(np.asarray(x['offsets'])) if 'offsets' in x else np.array([len(x['val'])])
        for x in tokens])

    tokens_cat = {
        k: torch.cat([torch.as_tensor(x[k]) for x in tokens])
        for k in tokens[0].keys()
    }

    offsets = torch.from_numpy(_counts_to_offset(counts))

    return {
        **tokens_cat,
        'offsets': offsets
    }


class CachedPrimitiveDataset(torch.utils.data.Dataset[SketchTokenSet]):
    """This dataset encapsulates a set of pre-tokenized sequences.

    This dataset presents functionality similar to `PrimitiveDataset`, but instead
    of tokenizing sequences on the fly, it relies on a pre-computed sequence file.
    """
    _offsets: torch.Tensor
    _val_tokens: torch.Tensor
    _pos_tokens: torch.Tensor
    _coord_tokens: torch.Tensor

    def __init__(self, cached_sequence_file: str, num_bins: int, max_length: Optional[int]=None):
        super().__init__()

        data = np.load(cached_sequence_file)

        if data['num_bins'] != num_bins:
            raise ValueError('Specified number of bins does not match number of bins in cached sequence file.')

        self._offsets = torch.from_numpy(data['offsets'])
        self._val_tokens = torch.from_numpy(data['val'])
        self._pos_tokens = torch.from_numpy(data['pos'])
        self._coord_tokens = torch.from_numpy(data['coord'])

        self.num_bins = num_bins
        self.max_length = max_length

    def __getitem__(self, idx: int) -> SketchTokenSet:
        idx_start = self._offsets[idx]
        idx_stop = self._offsets[idx + 1]
        segment = slice(idx_start, idx_stop)

        result = {
            'val': self._val_tokens[segment],
            'pos': self._pos_tokens[segment],
            'coord': self._coord_tokens[segment]
        }

        # Pytorch much prefers 64-bit integer indexing for Embeddings.
        return {
            k: _pad_or_truncate_to_length(v.to(dtype=torch.int64), self.max_length) for k, v in result.items()
        }

    def __len__(self) -> int:
        return len(self._offsets) - 1


class ImageTokenDatum(TypedDict):
    img: torch.Tensor
    val: torch.Tensor
    coord: torch.Tensor
    pos: torch.Tensor


class ImagePrimitiveDataset(torch.utils.data.Dataset[ImageTokenDatum]):
    """This class encapsulates a dataset of sequences, as well as accompanying renderings
    of corresponding sketches.

    """
    def __init__(self, primitive_dataset : PrimitiveDataset, image_dataset: MultiImageDataset,
                 image_transform: Optional[Callable[[PIL.Image.Image], PIL.Image.Image]]=None,
                 use_noisy_images: bool=True,
                 generator: torch.Generator=None):
        """Creates a new dataset from the given data.

        Parameters
        ----------
        primitive_dataset : PrimitiveDataset
            Path to the file containing the sequence data.
        image_dataset : MultiImageDataset
            Dataset containing the image data, with potentially multiple images per rendering.
        image_transform : Callable[[Image], Image], optional
            A callable transform to apply to the image, if provided.
        use_noisy_images : bool
            If `True`, indicates that noisy images from the image dataset are to be used.
            Otherwise, indicates that we use reference renderings.
        """
        self.sequences = primitive_dataset
        self.images = image_dataset
        self._idx = None
        self.generator = generator

        self.num_bins = primitive_dataset.num_bins
        self.max_length = primitive_dataset.max_length

        self.image_transform = image_transform
        self.use_noisy_images = use_noisy_images

        if len(self.images) != len(self.sequences):
            logging.getLogger(__name__).info('Number of images does not match number of sequences. Subsetting sequences to match images.')
            self._idx = self.images.indexes

    def __getitem__(self, idx) -> ImageTokenDatum:
        if self._idx is not None:
            idx = self._idx[idx]

        sample = self.sequences[idx]

        images = self.images[idx]
        if self.use_noisy_images and len(images) > 1:
            # This is the general case where we have noisy samples, we sample one noisy sample.
            image_bytes = images[torch.randint(1, len(images), (), generator=self.generator)]
        else:
            # In this case, we only have one rendering per sequence, which is the standard one.
            # Simply use that one if it's the only one we have.
            image_bytes = images[0]

        try:
            img = PIL.Image.open(io.BytesIO(image_bytes))
            if self.image_transform is not None:
                img = self.image_transform(img)

            img_tensor = torch.from_numpy(np.asarray(img).copy()).unsqueeze_(0).to(torch.float32).div_(255).sub_(0.5).mul_(2)
        except PIL.UnidentifiedImageError:
            logging.getLogger(__name__).warn('Failed to decode image at index {}'.format(idx))
            img_tensor = torch.zeros((1, 64, 64))

        return {
            'img': img_tensor,
            **sample
        }

    def __len__(self):
        return len(self._idx) if self._idx is not None else len(self.sequences)


def process_sequence_data(dataset: PrimitiveDataset, num_workers: Optional[int]=None) -> Dict[str, torch.Tensor]:
    """Tokenizes all the data from the given `PrimitevDataset`.

    Parameters
    ----------
    dataset : PrimitiveDataset
        The dataset from which to tokenize the data.
    num_workers : optional, int
        The number of workes to use when processing the data.
        If `None`, uses all available CPUs on the current machine.
    """
    if num_workers is None:
        num_workers = len(os.sched_getaffinity(0))

    dataloader = torch.utils.data.dataloader.DataLoader(
        dataset, 512,
        num_workers=num_workers,
        collate_fn=concatenate_tokens)

    results = []

    for batch in tqdm.tqdm(dataloader, smoothing=0.01):
        batch = {k: v.clone() for k, v in batch.items()}
        for k in ('val', 'coord', 'pos'):
            batch[k] = batch[k].to(dtype=torch.int16)
        results.append(batch)

    result = concatenate_tokens(results)
    result['num_bins'] = dataset.num_bins

    return result
