"""This script can be used to execute the end-to-end Vitruvion pipeline: 
sampling from the raw generative model, then sampling from the constraint 
model, and finally solving the sketch via Onshape. 

code-block:: bash 

    python -m img2cad.sampling_utils primitives.model_checkpoint=$primitive_path constraints.model_checkpoint=$constraint_path

"""

import dataclasses 
import glob 
import logging 
import omegaconf
import os 
import pickle
from typing import Optional, List, Dict

import hydra 
from hydra.core.config_store import ConfigStore 
import matplotlib.pyplot as plt 
import numpy as np 
import PIL 
import torch 
import tqdm 

from img2cad.dataset import sketch_from_tokens
from img2cad.sample_img2prim import sample as sample_img2prim
from img2cad.evaluation.sample_constraints import draw_samples as constraint_sample
from img2cad import primitives_models 
from img2cad.constraint_models import ConstraintModule
from img2cad.pipeline.prerender_images import render_sketch
import sketchgraphs
from sketchgraphs.data.sketch import Sketch 
import sketchgraphs.onshape.call as onshape_call

# --- env 
log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

@dataclasses.dataclass 
class OnshapeConfig: 
    credentials_path: str = omegaconf.MISSING
    document_url: str = omegaconf.MISSING

@dataclasses.dataclass 
class PrimitivesSamplingConfig: 
    model_checkpoint: str = omegaconf.MISSING
    conditional: Optional[bool] = False 
    image_dir: Optional[str] = None 
    num_samples: Optional[int] = 1
    max_token_len: Optional[int] = 130 

@dataclasses.dataclass 
class ConstraintSamplingConfig: 
    model_checkpoint: str = omegaconf.MISSING
    num_samples: Optional[int] = 1
    max_token_len: Optional[int] = 130 
    top_p: Optional[float] = 0.7

@dataclasses.dataclass 
class EndToEndConfig: 
    """Configuration for end-to-end sampling.

    Attributes
    ----------
    device : str, optional
        If not None, the name of the pytorch device to use. Otherwise, uses cuda if available,
        and cpu otherwise.
    """

    primitives: Optional[PrimitivesSamplingConfig] = dataclasses.field(default_factory=PrimitivesSamplingConfig)
    constraints: Optional[ConstraintSamplingConfig] = dataclasses.field(default_factory=ConstraintSamplingConfig)
    onshape: Optional[OnshapeConfig] = dataclasses.field(default_factory=OnshapeConfig)

    device: Optional[str] = None
    render: Optional[bool] = True 
    dpi: Optional[int] = 128 

def configure_onshape(config: OnshapeConfig): 
    creds_path_abs = hydra.utils.to_absolute_path(config.credentials_path)

    # --- confirm that the creds.json exists 
    if os.path.exists(creds_path_abs) is False: 
        log.error(f"Did not find a creds.json at {creds_path_abs}!")
        raise(FileNotFoundError) 

    # --- verify version identifiers of feature_template.json 
    try: 
        onshape_call.update_template(config.document_url)
    except Exception as e: 
        log.error("unable to update feature_template.json identifiers!")
        raise(e)

    log.info("Onshape configured successfully, solving")

def get_conditional_image(path: os.PathLike) -> torch.Tensor: 
    cond_img = PIL.Image.open(path)
    cond_img = torch.from_numpy(np.asarray(cond_img)[:,:,0].copy()).unsqueeze_(0).to(torch.float32).div_(255).sub_(0.5).mul_(2)
    return cond_img

def sample_primitives(config: PrimitivesSamplingConfig, device: torch.device) -> List[Sketch]: 
    checkpoint_path_abs = hydra.utils.to_absolute_path(config.model_checkpoint)

    if config.conditional: 
        module = primitives_models.ImagePrimitiveModule.load_from_checkpoint(checkpoint_path_abs)
    else: 
        module = primitives_models.RawPrimitiveModule.load_from_checkpoint(checkpoint_path_abs)

    model: torch.nn.Module = module.model
    hparams = module.hparams

    model = model.eval().to(device)
    log.info(f"Loaded primitive model to {device}... sampling.")

    samples = [] 
    if config.conditional: 
        image_dir_abs = hydra.utils.to_absolute_path(config.image_dir)
        cond_img_paths = glob.glob(os.path.join(image_dir_abs, "*.png"))
        config.num_samples = min(config.num_samples, len(cond_img_paths))


    for i in tqdm.trange(config.num_samples, desc="Primitives sampling", smoothing=0.01):
        if config.conditional: 
            cond_img = get_conditional_image(cond_img_paths[i])
        else:
            cond_img = None

        tokens = sample_img2prim(model, config.max_token_len, cond_img=cond_img, device=device)
        sketch = sketch_from_tokens(tokens, hparams["data"]["num_position_bins"])
        samples.append(sketch)

    return samples 

def sample_constraints(config: ConstraintSamplingConfig, primitives: List[Sketch], device: torch.device) -> List[List[Sketch]]: 
    checkpoint_path_abs = hydra.utils.to_absolute_path(config.model_checkpoint)

    model = ConstraintModule.load_from_checkpoint(checkpoint_path_abs)
    model = model.eval().to(device)

    log.info(f"Loaded constraint model to {device}... sampling.")

    samples = [] 
    skipped = 0 

    for i, sketch in enumerate(tqdm.tqdm(primitives, desc="Constraint sampling", smoothing=0.01)):
        try: 
            constrained_sketch = constraint_sample(model, sketch, max_token_len=config.max_token_len, top_p=config.top_p, num_samples=config.num_samples, progress=False, device=device)
            samples.append(constrained_sketch)
        except: 
            log.warning(f"Unable to constrain sketch {i}! Skipping...")
            samples.append(None)
            skipped += 1

    if skipped: 
        log.warning(f"Skipped {skipped} of {len(primitives)} sketches...")

    return samples 

def solve_onshape(config: OnshapeConfig, constraints: List[List[Sketch]]) -> Dict[str, Sketch]: 
    solved = {} 

    for i in tqdm.tqdm(range(len(constraints)), desc="Onshape solving"): 
        for j, sketch in enumerate(constraints[i]): 
            name = f"solved_{i:03d}_{j:03d}"

            onshape_call.add_feature(config.document_url, sketch.to_dict(), sketch_name=name)
            solved_sketch_info = onshape_call.get_info(config.document_url, name)
            solved_sketch = sketchgraphs.data.Sketch.from_info(solved_sketch_info['geomEntities'])
            solved[name] = solved_sketch

    return solved 

@hydra.main(config_name="config")
def main(config: EndToEndConfig): 
    render_dir = 'renders'

    device = config.device
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)

    if not config.primitives:
        return

    # --- sample from the primitive model 
    primitives = sample_primitives(config.primitives, device)

    primitives_save_path = os.path.abspath('primitives.pkl')
    log.info(f'Saving sampled primitives at {primitives_save_path}')
    with open(primitives_save_path, 'wb') as f:
        pickle.dump(primitives, f)

    if config.render: 
        os.makedirs(render_dir, exist_ok=True)

        for i, sketch in enumerate(tqdm.tqdm(primitives, desc="Primitives renders")):
            path = os.path.join(render_dir, f"primitives_{i:03d}.pdf")
            fig = render_sketch(sketch, return_fig=True)
            fig.savefig(path, dpi=config.dpi)
            plt.close()

    # --- sample constraints 
    if not config.constraints: 
        return

    constraints = sample_constraints(config.constraints, primitives, device)

    constraints_save_path = os.path.abspath('constraints.pkl')
    log.info(f'Saving sampled constraints at {constraints_save_path}')
    with open(constraints_save_path, 'wb') as f:
        pickle.dump(constraints, f)

    if config.render:
        for i in tqdm.tqdm(range(len(constraints)), desc="Contraint renders"): 
            for j, sketch in enumerate(constraints[i]): 
                path = os.path.join(render_dir, f"constraints_{i:03d}_{j:03d}.pdf")
                fig = render_sketch(sketch, return_fig=True)
                fig.savefig(path, dpi=config.dpi)
                plt.close()

    # --- solve via onshape 
    if not config.onshape: 
        log.info('Skipping onshape solving as not requested.')
        return

    configure_onshape(config.onshape)
    solved = solve_onshape(config.onshape, constraints)

    if config.render: 
        for name, sketch in solved.items(): 
            path = os.path.join(render_dir, f"{name}.pdf")
            fig = render_sketch(sketch, return_fig=True)
            fig.savefig(path, dpi=config.dpi)
            plt.close()

if __name__=="__main__": 
    cs = ConfigStore.instance() 
    cs.store(name="config", node=EndToEndConfig)
    main() 
