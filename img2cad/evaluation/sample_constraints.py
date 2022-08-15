"""This script samples from a trained constraint generation model.
"""

import copy
import dataclasses
import logging
import gzip
import os
import pickle

import hydra
import omegaconf
import torch
import torch.utils.data
import torch.nn.functional as F
import tqdm

from typing import Dict, List, Optional

from img2cad import constraint_data, data_utils, constraint_models, primitives_data
from img2cad.dataset import tokenize_sketch
from img2cad.sample_img2prim import top_p_filtering

from sketchgraphs import data as datalib


def sketch_from_tokens(tokens, sketch: datalib.Sketch, gather_idxs):
    """Add constraint value tokens to Sketch instance.
    """
    seq = datalib.sketch_to_sequence(sketch)
    stop_node = seq.pop()  # add back afterwards

    reverse_gather = {val:idx for idx, val in enumerate(gather_idxs)}
    new_type = None
    new_refs = []
    for token in tokens:
        if token < len(constraint_data.Token):
            # Add previous completed constraint, if any
            if new_type is not None:
                new_op = datalib.EdgeOp(
                    label=new_type, references=tuple(new_refs))
                new_type = None
                new_refs = []
                seq.append(new_op)
            if token <= constraint_data.Token.Stop:
                continue
            # Get constraint type
            new_type = datalib.ConstraintType[constraint_data.Token(token).name]
        else:
            ori_token = token - len(constraint_data.Token)
            new_refs.append(reverse_gather[ori_token])

    seq.append(stop_node)
    return datalib.sketch_from_sequence(seq)

def sample(model: constraint_models.ConstraintModule, max_len: int, top_p: float=0.9, temperature: float=1.0,
           priming_toks: Dict[str, torch.Tensor]=None, device: torch.device=None):
    """Draw a sample from the given model.

    Parameters
    ----------
    model : ConstraintModule
        The model to draw a sample from
    max_len : int
        Maximum length of sampled token sequence
    top_p : float
        Minimum cumulative probability for nucleus sampling
    temperature : float
        Softmax temperature
    priming_toks : dict
        A dictionary of val, coord, and pos tokens to condition on (primitives)

    Returns
    -------
    list
        The sampled value tokens
    """
    if device is None:
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    model = model.eval()

    num_out = len(constraint_data.Token) + max_len
    def sample_bounded(tok_input, min_idx, max_idx):
        tok_input = copy.deepcopy(tok_input)
        for tok_type, tokens in tok_input.items():
            if not isinstance(tokens, torch.Tensor):
                tokens = torch.tensor(tokens)
            tok_input[tok_type] = tokens[None, ...].to(device=device)

        output = model(tok_input)
        logits = output[0, -1, min_idx:max_idx+1]
        # Nucleus sampling w/ temperature
        logits = logits / temperature
        filtered_logits = top_p_filtering(logits, top_p)
        probs = F.softmax(filtered_logits, dim=-1)
        pred = torch.multinomial(probs, 1)
        return (pred.item() + min_idx)

    with torch.no_grad():
        tok_input = copy.deepcopy(priming_toks)
        tok_input['c_val'] = [constraint_data.Token.Start]
        tok_input['c_coord'] = [constraint_data.NON_COORD_TOKEN]
        pos_idx = 1
        tok_input['c_pos'] = [pos_idx]
        while len(tok_input['c_val']) < max_len:
            pred = sample_bounded(tok_input, constraint_data.Token.Stop, len(constraint_data.Token)-1)
            if pred == constraint_data.Token.Stop:
                tok_input['c_val'].append(constraint_data.Token.Stop)
                break
            tok_input['c_val'].append(pred)
            tok_input['c_coord'].append(constraint_data.NON_COORD_TOKEN)
            pos_idx += 1
            tok_input['c_pos'].append(pos_idx)
            for coord_tok in constraint_data.CONSTRAINT_COORD_TOKENS:
                if coord_tok == 2:  # fix below
                    pred = sample_bounded(tok_input, len(constraint_data.Token)+1, num_out-1)
                else:
                    pred = sample_bounded(tok_input, constraint_data.Token.Stop, num_out-1)
                if pred >= len(constraint_data.Token):
                    tok_input['c_val'].append(pred)
                    tok_input['c_coord'].append(coord_tok)
                    tok_input['c_pos'].append(pos_idx)
    sampled_tokens = tok_input['c_val']
    return sampled_tokens


def draw_samples(model: constraint_models.ConstraintModule, priming_sketch: datalib.Sketch, 
                 max_token_len: int, top_p: float, num_samples: int=20, progress: bool=True,
                 device: Optional[torch.device]=None) -> List[datalib.Sketch]:
    
    # make sure constraints are not already populated 
    if priming_sketch.constraints: 
        raise ValueError("This sampling utility assumes that the priming sketch is unconstrained")
    
    priming_toks, gather_idxs = tokenize_sketch(
        priming_sketch,
        model.hparams.data.num_position_bins,
        model.hparams.data.max_token_length)

    all_samples = []

    for _ in tqdm.trange(1, num_samples+1, disable=not progress):
        this_sample = sample(model, max_token_len, top_p, priming_toks=priming_toks, device=device)
        sketch = sketch_from_tokens(this_sample, priming_sketch, gather_idxs)
        all_samples.append(sketch)

    return all_samples

def load_sketch_from_onshape(url: str) -> datalib.Sketch:
    """Loads sketch from the given URL.
    """
    import sketchgraphs.onshape.call as onshape_call

    sketch_info = onshape_call.get_info(url)
    sketch_names = [sk['sketch'] for sk in sketch_info['sketches']]
    input_sk_name = 'Input'
    if sketch_names.count(input_sk_name) != 1:
        raise ValueError("Exactly 1 sketch must be named 'Input'")
    sk_idx = sketch_names.index(input_sk_name)
    sketch_info = sketch_info['sketches'][sk_idx]
    sketch = datalib.Sketch.from_info(sketch_info['geomEntities'])
    data_utils.normalize_sketch(sketch)
    return sketch

def store_sketches_to_onshape(url: str, sketches: List[datalib.Sketch]):
    """Stores sketches to the given URL.
    """

    import sketchgraphs.onshape.call as onshape_call

    # Send to Onshape
    for sk_idx, sketch in enumerate(sketches):
        onshape_call.add_feature(url, sketch.to_dict(), sketch_name='Sample %i' % sk_idx)

def load_model(checkpoint_path: str) -> constraint_models.ConstraintModule:
    model = constraint_models.ConstraintModule.load_from_checkpoint(checkpoint_path, map_location='cpu')
    return model


@dataclasses.dataclass
class SampleConstraintConfig:
    checkpoint_path: str = omegaconf.MISSING
    url: Optional[str] = None
    limit_sketches: Optional[int] = None
    sequence_path: Optional[str] = None


def sample_constraints_from_url(model: constraint_models.ConstraintModule, url: str):
    logging.info(f'Loading sketch from Onshape at {url}')
    sketch = load_sketch_from_onshape(url)

    logging.info('Drawing constraint samples')
    samples = draw_samples(model, sketch, model.hparams.data.max_token_length, 0.9)

    logging.info(f'Storing sampled sketches to Onshape at {url}')
    store_sketches_to_onshape(url, samples)


def sample_constraints_from_dataset(model: constraint_models.ConstraintModule, dataset: constraint_data.ConstraintDataset):
    all_samples = []

    for i in tqdm.trange(len(dataset)):
        sketch = dataset[i]
        samples = draw_samples(model, sketch, model.hparams.data.max_token_length, 0.7, num_samples=5, progress=False)
        all_samples.append(samples)

    return all_samples


def sample_and_save_constraints_from_dataset(model: constraint_models.ConstraintModule, config: SampleConstraintConfig):

    if config.sequence_path is not None:
        # Override hparams stored sequence path
        sequence_path = hydra.utils.to_absolute_path(config.sequence_path)
    else:
        sequence_path = model.hparams.data.sequence_path

    dataset = constraint_data.ConstraintDataset(
        sequence_path,
        model.hparams.data.num_position_bins,
        model.hparams.data.max_token_length,
        tokenize=False,
        primitive_noise_config=constraint_data.PrimitiveNoiseConfig(enabled=True))

    _, _, dataset = primitives_data.split_dataset(
        dataset, model.hparams.data.validation_fraction, model.hparams.data.test_fraction)

    if config.limit_sketches is not None and len(dataset) > config.limit_sketches:
        dataset = torch.utils.data.Subset(dataset, range(config.limit_sketches))

    result = sample_constraints_from_dataset(model, dataset)

    output_path = os.path.abspath('result.pkl.gz')
    logging.getLogger(__name__).info(f'Saving inferred constraints to path {output_path}.')

    with gzip.open(output_path, 'wb') as f:
        pickle.dump(result, f, protocol=4)


@hydra.main(config_name='conf')
def main(config: SampleConstraintConfig):
    logger = logging.getLogger(__name__)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = load_model(hydra.utils.to_absolute_path(config.checkpoint_path))
    model = model.to(device=device)

    if config.url is not None:
        logger.info('Fetching sketch and completing constraints from URL.')
        sample_constraints_from_url(model, config.url)
        return

    sample_and_save_constraints_from_dataset(model, config)


if __name__ == '__main__':
    from hydra.core.config_store import ConfigStore
    cs = ConfigStore()
    cs.store('conf', node=SampleConstraintConfig)
    main()
