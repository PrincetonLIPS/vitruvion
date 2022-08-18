"""This module can be used to reproduce the figures appearing in 
the Vitruvion supplementary material. 
"""

import dataclasses
from functools import reduce 
import logging
import os 
from typing import List, Tuple, Optional

import hydra 
from hydra.core.config_store import ConfigStore 
import matplotlib.pyplot as plt 
from matplotlib.pyplot import Figure
import omegaconf
import torch 

from img2cad.pipeline.prerender_images import render_sketch
import img2cad.sampling_utils as sampling_utils
from sketchgraphs.data.sketch import Sketch 

# --- env 
log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

@dataclasses.dataclass
class FigureGenerationConfig: 
    """This class encapsulates the configuration for figure generation. 
    """
    output: str = omegaconf.MISSING
    unconditional_grid: Optional[bool] = False 
    conditional_grid: Optional[bool] = True 
    primitive_checkpoint: Optional[str] = None
    primitive_samples: Optional[int] = 1
    primitive_img_dir: Optional[str] = None 

def setup_output_dir(config: FigureGenerationConfig) -> None: 
    output_abs = hydra.utils.to_absolute_path(config.output)

    if os.path.exists(output_abs) is False: 
        log.info(f"Figure directory {config.output} not found... creating it.")
        os.mkdir(output_abs)

    log.info(f"Writing output to {output_abs}")

def make_grid(sketches: List[Sketch], grid_size: Tuple[int, int]) -> Figure: 
    rows, cols = grid_size 
    assert len(sketches) == (rows * cols), "grid size doesn't match the number of sketches provided!"

    fig, axs = plt.subplots(nrows=rows, ncols=cols) 
    for ax, sketch in zip(axs.reshape(-1), sketches): 
        _ = render_sketch(sketch, ax=ax, return_fig=True) 
        ax.set(aspect="equal")

    return fig 

@hydra.main(config_name="config")
def main(config: FigureGenerationConfig): 
    setup_output_dir(config)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if config.unconditional_grid: 
        primitive_config = sampling_utils.PrimitivesSamplingConfig(model_checkpoint=config.primitive_checkpoint, num_samples=49) 
        samples = sampling_utils.sample_primitives(primitive_config, device)
        fig = make_grid(samples, (7, 7))
        fig.savefig(os.path.join(hydra.utils.to_absolute_path(config.output), "unconditional_grid.pdf"))
        plt.close() 
        log.info("Rendered unconditional grid figure")

    if config.conditional_grid: 
        grid_size = (9, 4)
        primitive_config = sampling_utils.PrimitivesSamplingConfig(model_checkpoint=config.primitive_checkpoint, conditional=True, image_dir=config.primitive_img_dir, num_samples=4)
        samples = sampling_utils.sample_primitives(primitive_config, device)
        fig = make_grid(samples, grid_size)
        fig.savefig(os.path.join(hydra.utils.to_absolute_path(config.output), "conditional_grid.pdf"))
        plt.close() 
        log.info("Rendered conditional grid figure")

if __name__=="__main__": 
    cs = ConfigStore.instance() 
    cs.store(name="config", node=FigureGenerationConfig)

    main() 
