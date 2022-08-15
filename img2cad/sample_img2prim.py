"""This script samples from a trained primitive generation model.
"""

import argparse
import copy
import os
from typing import Dict, Sequence

from tqdm import tqdm
import numpy as np
import PIL, PIL.Image
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F

import sketchgraphs.data as datalib
from . import dataset
from .dataset import Token
from . import primitives_models, primitives_data
from .pipeline.prerender_images import render_sketch


def top_p_filtering(logits, top_p=0.9, filter_value=-float('Inf')):
    """Filter a distribution of logits using top-p/nucleus filtering.

    Originally from:
    https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317

    Parameters
    ----------
    logits : torch.tensor
        The logits of the distribution
    top_p : float
        Minimum cumulative probability
    filter_value : float
        Value to set the ignored elements to

    Returns
    -------
    torch.tensor
        Modified logits with ignored elements set to filter_value
    """
    assert logits.dim() == 1  # batch size 1 for now
    assert 0 < top_p and top_p <= 1.0
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
    # Remove tokens with cumulative probability above the threshold
    sorted_indices_to_remove = cumulative_probs >= top_p
    # Shift the indices to the right to keep first above-threshold token
    sorted_indices_to_remove[..., 1:] = (
        sorted_indices_to_remove[..., :-1].clone())
    sorted_indices_to_remove[..., 0] = 0
    # Filter logits
    indices_to_remove = sorted_indices[sorted_indices_to_remove]
    logits[indices_to_remove] = filter_value
    return logits


def sample(model, max_len: int, top_p: float=0.9, temperature: float=1.0, cond_img: torch.Tensor=None,
           tok_input: Dict[str, Sequence[int]]=None, device: torch.device=None):
    """Draw a sample from the given model.

    Parameters
    ----------
    model : ImageToPrimitiveModel
        The model to draw a sample from
    max_len : int
        Maximum length of sampled token sequence
    top_p : float
        Minimum cumulative probability for nucleus sampling
    temperature : float
        Softmax temperature
    cond_img : torch.tensor
        An image to condition on

    Returns
    -------
    list
        The sampled value tokens
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model.eval()
    num_out = model.out.out_features
    max_entities = model.pos_embed.weight.shape[0] - 3

    def sample_bounded(tok_input: Dict[str, Sequence[int]], min_idx, max_idx):
        tok_input = copy.deepcopy(tok_input)
        for tok_type, tokens in tok_input.items():
            if not isinstance(tokens, torch.Tensor):
                tokens = torch.tensor(tokens)
            # Add singleton batch dimension
            tok_input[tok_type] = tokens[None, ...].to(device)
        output = model(tok_input)
        logits = output[0, -1, min_idx:max_idx+1]
        # Nucleus sampling w/ temperature
        logits = logits / temperature
        filtered_logits = top_p_filtering(logits, top_p)
        probs = F.softmax(filtered_logits, dim=-1)
        pred = torch.multinomial(probs, 1)
        return (pred.item() + min_idx)

    if tok_input is None:
        pos_idx = 1  # TODO dry-ify tokenization
        tok_input = {'val': [Token.Start],
                    'coord': [dataset.NON_COORD_TOKEN],
                    'pos': [pos_idx]}
    else:
        pos_idx = tok_input['pos'][-1] + 1
        tok_input = {
            k: list(v) for k, v in tok_input.items()
        }

    with torch.no_grad():
        if cond_img is not None:
            tok_input['img'] = cond_img
        while len(tok_input['val']) < max_len:
            pred = sample_bounded(tok_input, Token.Stop, len(Token)-1)
            # Check if max number of ents reached already
            if pos_idx == 1 + max_entities:
                pred = Token.Stop
            if pred == Token.Stop:
                tok_input['val'].append(Token.Stop)
                break
            tok_input['val'].append(pred)
            tok_input['coord'].append(dataset.NON_COORD_TOKEN)
            pos_idx += 1
            tok_input['pos'].append(pos_idx)
            ent_class = datalib.ENTITY_TYPE_TO_CLASS[
                datalib.EntityType[Token(pred).name]]  # TODO: clean up
            for coord_tok in dataset.COORD_TOKEN_MAP[ent_class]:
                if dataset.INCLUDE_CONSTRUCTION:
                    max_coord_idx = num_out - 3
                else:
                    max_coord_idx = num_out - 1
                pred = sample_bounded(tok_input, len(Token), max_coord_idx)
                tok_input['val'].append(pred)
                tok_input['coord'].append(coord_tok)
                tok_input['pos'].append(pos_idx)

            # isConstruction attribute
            if dataset.INCLUDE_CONSTRUCTION:
                pred = sample_bounded(tok_input,
                    max_coord_idx+1, max_coord_idx+2)
                tok_input['val'].append(pred)
                tok_input['coord'].append(dataset.NON_COORD_TOKEN)
                tok_input['pos'].append(pos_idx)

    sampled_tokens = tok_input['val']
    return sampled_tokens


def draw_samples(model, save_dir, num_bins, max_token_len, top_p,
                 cond_img, num_samples=20):
    all_samples = []
    for samp_idx in tqdm(range(1, num_samples+1)):
        this_sample = sample(model, max_token_len, top_p, cond_img=cond_img)
        sketch = dataset.sketch_from_tokens(this_sample, num_bins)
        all_samples.append(sketch)
    return all_samples


def main():
    parser = argparse.ArgumentParser(
        description='Sample of image-conditional primitive model')
    parser.add_argument('--model_path', type=str,
        help='Path to saved model')
    parser.add_argument('--save_dir', type=str,
        help='Path to directory for sample saving')
    parser.add_argument('--num_samples', type=int, default=10,
        help='Number of samples per conditioning image')
    parser.add_argument('--top_p', type=float, default=0.9,
        help='Minimum cumulative probability for nucleus sampling')
    parser.add_argument('--num_images', type=int, default=10,
        help='Number of images to condition on (they are randomly selected')
    parser.add_argument('--partition', type=str, default='val',
        choices=['train', 'val'], help='Partition to select cond. images from')
    
    args = parser.parse_args()

    # Create save directory
    os.makedirs(args.save_dir)

    # Load model
    pl_module = primitives_models.ImagePrimitiveModule.load_from_checkpoint(
        args.model_path, map_location='cpu')
    model = pl_module.model
    model = model.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    hparams = pl_module.hparams

    # Get dataset
    data_config = primitives_data.ImagePrimitiveDataConfig()
    # data_config.sequence_path = hparams['data']['sequence_path']
    # data_config.image_data_folder = hparams['data']['image_data_folder']
    # data_config.sequence_path = '/home/wenda/gencad/code/data/sg_filtered_unique.npy'
    data_config.sequence_path = '../data/sequence_data/filtered_unique/sg_filtered_unique.npy'
    # data_config.image_data_folder = '128_shard'
    data_config.image_data_folder = '/home/nrichardson/renders_128p'
    pl_data = primitives_data.ImagePrimitiveDataModule(data_config)
    pl_data.prepare_data()
    pl_data.setup()

    # Sample (condition on sharded images)
    for idx in tqdm(range(args.num_images)):
        # Create example-specific directory
        this_savedir = os.path.join(args.save_dir, '%03i' % idx)
        os.makedirs(this_savedir)

        # Get image for conditioning
        if args.partition == 'val':
            cond_dataset = pl_data._valid_dataset
        else:
            cond_dataset = pl_data._train_dataset
        img_idx = np.random.choice(len(cond_dataset))

        with open(os.path.join(this_savedir, 'info.txt'), 'w') as fh:
            fh.write('Image index: ' + str(img_idx))
        cond_img = cond_dataset[img_idx]['img']

        # #################################
        # cond_img = PIL.Image.open('hand_drawn_playing/rpa_permute/67.png')
        # cond_img = PIL.Image.open('hand_drawn_playing/rpa_permute/45.png')
        # # cond_img = PIL.Image.open('hand_drawn_playing/rpa_64251/45.png')
        # # # cond_img = PIL.Image.open('new_noise2.png')
        # # # cond_img = PIL.Image.open('validation_samples/001/cond.png')
        # # # cond_img = cond_img.transpose(PIL.Image.FLIP_LEFT_RIGHT)
        cond_img = PIL.Image.open('../data/hand_drawn/manually_drawn13.png')
        # # # # cond_img = PIL.Image.open('cond.png')

        cond_img = torch.from_numpy(
            np.asarray(cond_img)[:,:,0].copy()).unsqueeze_(0).to(
                torch.float32).div_(255).sub_(0.5).mul_(2)

        # # LR flip
        # cond_img = torch.flip(cond_img, [2])

        # # UD flip
        # cond_img = torch.flip(cond_img, [1])

        # # Transpose
        # cond_img = torch.transpose(cond_img, 1, 2)

        # Translate to the right
        # trans_dist = 6
        # cond_img[0, :, trans_dist:] = cond_img[0, :, :-trans_dist].clone()
        # cond_img[0, :, :trans_dist] = 1.

        # Translate to the left
        # trans_dist = 5
        # cond_img[0, :, :-trans_dist] = cond_img[0, :, trans_dist:].clone()
        # cond_img[0, :, -trans_dist:] = 1.
        #################################

        plt.imsave(os.path.join(this_savedir, 'cond.png'),
            cond_img.numpy()[0], cmap='gray')

        # Draw samples
        sampled_sk = draw_samples(model, args.save_dir, 
            hparams['data']['num_position_bins'],
            hparams['data']['max_token_length'],
            args.top_p, cond_img, num_samples=args.num_samples)

        # Render samples
        for sk_idx, sketch in enumerate(sampled_sk):
            fig = render_sketch(sketch, return_fig=True)
            fig.savefig(os.path.join(this_savedir, '%03i.pdf' % sk_idx),
                dpi=128)
            plt.close()


if __name__ == '__main__':
    main()
