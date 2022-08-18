"""This script pre-renders and saves SG images.

This script processes the given sketches to render sketches.

.. code-block:: bash
    
    # --- pralexa 
    python -m img2cad.pipeline.prerender_images sequence_file=/home/wenda/gencad/code/data/sg_filtered_unique.npy

    # --- wash 
    python -m img2cad.pipeline.prerender_images sequence_file=/n/fs/sketchgraphs/unique/sg_filtered_unique.npy

To render the sketches by deploying a slurm array job, provide the current 
array task id and the task count, see `img2cad.pipeline.render_images.sh` for details. 

To obtain filtered sequences, see `img2cad.pipeline.filter_sequences`.

"""

import dataclasses
import functools
import io
import multiprocessing
import os
from copy import deepcopy

from typing import List, Optional, Tuple 

import hydra
import numpy as np
import matplotlib.pyplot as plt
import omegaconf
import threadpoolctl
import tqdm
import PIL.Image

import sketchgraphs.data as datalib
from sketchgraphs.data import flat_array
from img2cad import data_utils as i2c_utils
from img2cad import plotting



def render_sketch(sketch: datalib.Sketch, ax=None, pad: float=0.1, size_pixels: int=128, sketch_extent: Optional[float]=None, hand_drawn: bool=False, return_fig: bool=False) -> Optional[bytes]:
    """Renders the given sketch to a byte buffer.

    Parameters
    ----------
    sketch : datalib.Sketch
        The sketch to render.
    pad : float
        The amount to pad the sketch on each side by, as a fraction of the longest
        side of the unpadded sketch.
    size_pixels : int
        The size of the sketch to render, in pixels.
    sketch_extent : float, optional
        If not None, the size of the bounding box to use for the sketch. Otherwise, this
        is automatically deduced from the sketch.

    Returns
    -------
    bytes
        A byte array representing the PNG encoded image, if rendering was succesful,
        or `None` otherwise.
    """
    # Verify sketch is normalized
    if sketch_extent is None:
        (x0, y0), (x1, y1) = i2c_utils.get_sketch_bbox(sketch)
        w = x1 - x0
        h = y1 - y0
        sketch_extent = max(w, h)

    # Render sketch
    try:
        fig = plotting.render_sketch(sketch, ax=ax, hand_drawn=hand_drawn)
    except Exception as exc:
        import pickle
        import random
        import logging

        filename = os.path.abspath(f'error_sketch_{random.randint(0, 10000)}.pkl')
        logging.getLogger(__name__).error(f'Error processing sketch! Sketch dumped to {filename}')

        with open(filename, 'wb') as f:
            pickle.dump(sketch, f, protocol=4)
        raise

    # Adjust lims according to pad
    curr_lim = sketch_extent / 2
    new_lim = curr_lim + pad

    if ax is None: 
        fig.axes[0].set_xlim(-new_lim, new_lim)
        fig.axes[0].set_ylim(-new_lim, new_lim)
    else: 
        ax.set_xlim(-new_lim, new_lim)
        ax.set_ylim(-new_lim, new_lim)


    # Verify correct radii
    for ent in sketch.entities.values():
        if hasattr(ent, 'radius'):
            if getattr(ent, 'radius') == 0:
                return None

    if ax is None: 
        fig.set_size_inches(1, 1)
        fig.tight_layout() 

    if return_fig is True:
        return fig

    # Save image
    buffer = io.BytesIO()
    fig.savefig(buffer, dpi=size_pixels)
    plt.close(fig)

    buffer.seek(0)

    img = PIL.Image.open(buffer, formats=('PNG',))
    img = img.convert('L')

    output_buffer = io.BytesIO()
    img.save(output_buffer, format='PNG', optimize=True)
    return output_buffer.getvalue()


def process_sketch_sequence(sequence, num_noisy: int=0, pad: float=0.1, size_pixels: int=128) -> Optional[List[bytes]]:
    """Processes the given sequence of node / edge operations, rendering the sketch and an appropriate
    number of other variations.

    Parameters
    ----------
    sequence : Sequence
        A sketch represented as a sequence of operations.
    num_noisy : int
        The number of noisy versions to generate.
    pad : bool
        The padding to be applied, see `render_sketch`.
    size_pixels : int
        The size of the render, given as a pixel count for one edge.

    Returns
    -------
    List[bytes] or None
        If not None, a list of length ``num_noisy + 1``, representing renderings for the
        original sketch and noisy versions. If the normalization for the sketch failed,
        returns `None`.
    """
    sketch = datalib.sketch_from_sequence(sequence)
    scale_factor = i2c_utils.normalize_sketch(sketch)

    if scale_factor == -1:
        return None

    result = []
    result.append(render_sketch(sketch, pad, size_pixels, sketch_extent=1))

    for _ in range(num_noisy):
        sketch_noisy = deepcopy(sketch)
        # sketch_noisy = noise_models.RenderNoise(sketch_noisy)  # the rendering function in plotting does this
        result.append(render_sketch(sketch_noisy, pad, size_pixels, sketch_extent=1, hand_drawn=True))

    return result


def _process_sketch_sequence_raw(sequence_bytes: bytes, num_noisy: int, pad: float=0.1, size_pixels: int=128) -> Optional[List[bytes]]:
    """Wrapper for `process_sketch_sequence` which operates on raw representations of the data.

    This wrapper is used in multi-processing to avoid overly burdening the main thread with serializing / deserializing tasks.
    See `process_sketch_sequence` for descriptions of the method parameters.
    """
    with threadpoolctl.threadpool_limits(limits=1):
        sequence = flat_array.FlatSerializedArray.decode_raw_bytes(sequence_bytes)
        result = process_sketch_sequence(sequence, num_noisy, pad, size_pixels)

    if result is None:
        return None

    return [flat_array._save_single(r) for r in result]


@dataclasses.dataclass
class ImageRenderingConfiguration:
    """Configuration for rendering images from sketches.

    Attributes
    ----------
    sequence_file : str
        The input sequence file to process
    limit_sequences : int or None
        If not None, limit the number of processed sequences to that value.
    output_file : str
        The path of the output file to write. Note that if this path is
        relative, it will be relative to the output folder of the script.
    num_workers : int or None
        If not None, the number of CPU workers to use.
    image_size : int
        The size of the images to render in pixels.
    num_noisy_samples : int
        The number of noisy samples to generate per image.
    """
    sequence_file: str = omegaconf.MISSING
    slurm_array_task_id: Optional[int] = None
    slurm_array_task_count: Optional[int] = None 
    limit_sequences: Optional[int] = None
    output_file: str = 'images.npy'
    num_workers: Optional[int] = None
    image_size: int = 128
    num_noisy_samples: int = 0

def get_shard_range(shard_id: int, n_sequences: int, n_shards: int) -> Tuple[int, int]:
    remainder = n_sequences % n_shards
    shard_size = (n_sequences // n_shards) 
    shard_size_plus_one = shard_size + 1

    if shard_id < remainder: 
        i = shard_id * shard_size_plus_one
        j = i + shard_size_plus_one
    else: 
        i = remainder * shard_size_plus_one + (shard_id - remainder) * shard_size 
        j = i + shard_size
    return (i, j)

@hydra.main(config_name='conf')
def main(config: ImageRenderingConfiguration):
    sequence_path = hydra.utils.to_absolute_path(config.sequence_file)

    sequence_array: flat_array.FlatSerializedArray = flat_array.load_dictionary_flat(sequence_path)['sequences']
    num_sequences = len(sequence_array)
    shard_id = config.slurm_array_task_id 
    num_shards = config.slurm_array_task_count

    if config.limit_sequences is not None:
        num_sequences = min(num_sequences, config.limit_sequences)
        image_indices = range(num_sequences)
    elif config.slurm_array_task_id is not None: 
        i, j = get_shard_range(shard_id, num_sequences, num_shards)
        image_indices = range(i, j)
        num_sequences = min(num_sequences, j-i)
    else: 
        image_indices = range(num_sequences)

    sequence_bytes = (sequence_array.get_raw_bytes(i) for i in image_indices)

    process_fn = functools.partial(_process_sketch_sequence_raw, num_noisy=config.num_noisy_samples, size_pixels=config.image_size)

    if config.num_workers is None or config.num_workers > 0:
        pool = multiprocessing.Pool(config.num_workers)
        result = pool.imap(process_fn, sequence_bytes)
    else:
        pool = None
        result = map(process_fn, sequence_bytes)

    result = tqdm.tqdm(result, total=num_sequences, smoothing=0.01)

    buffer = io.BytesIO()
    offsets = []
    current_offset = 0

    num_skipped = 0
    indexes = []

    for i, r in zip(image_indices, result):
        if r is None:
            num_skipped += 1
            continue

        for img in r:
            buffer.write(img)
            offsets.append(current_offset)
            current_offset += len(img)
            indexes.append(i)

    if pool is not None:
        pool.close()

    offsets.append(current_offset)

    print('Done processing renders, skipped {} sequences'.format(num_skipped))

    offsets = np.array(offsets, dtype=np.int64)
    indexes = np.array(indexes, dtype=np.int64)

    imgs = flat_array.pack_list_flat(offsets, np.asarray(buffer.getbuffer()))
    result = flat_array.pack_dictionary_flat({
        'indexes': indexes,
        'imgs': imgs
    })

    output_path = os.path.abspath(config.output_file)
    print('Saving result to {}'.format(output_path))
    np.save(output_path, result)

if __name__ == '__main__':
    from hydra.core.config_store import ConfigStore
    cs = ConfigStore()
    cs.store('conf', node=ImageRenderingConfiguration)
    main()
