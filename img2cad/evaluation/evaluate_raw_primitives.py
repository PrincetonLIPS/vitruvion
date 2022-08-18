"""Script for evaluating raw primitive model."""


import dataclasses
import logging
import os
import pickle

from typing import Dict, Iterable, Optional, Tuple

import hydra
import numpy as np
import omegaconf
import torch
import torch.nn.functional as F
import torchmetrics
import tqdm

from img2cad import modules, primitives_data, primitives_models
from img2cad.dataset import Token

from sketchgraphs_models.nn.summary import CohenKappa


@dataclasses.dataclass
class RawPrimitiveEvaluationConfig:
    checkpoint_path: str = omegaconf.MISSING
    batch_size: int = 2048
    shuffle_primitives: Optional[bool] = None


def scatter_sum(src: torch.Tensor, index: torch.Tensor, dim: int = -1, dim_size: int=None) -> torch.Tensor:
    size = list(src.size())
    if dim_size is not None:
        size[dim] = dim_size
    elif index.numel() == 0:
        size[dim] = 0
    else:
        size[dim] = int(index.max()) + 1
    out = torch.zeros(size, dtype=src.dtype, device=src.device)
    return out.scatter_add_(dim, index, src)



def evaluate_model(model: primitives_models.RawPrimitiveModule, dataloader: Iterable[modules.TokenInput],
                   device: torch.device, num_positional_embeddings: int=None) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """Computes pre-primitive loss for the given model.

    Parameters
    ----------
    model : RawPrimitiveModule
        Model for which to compute the loss
    dataloader : Iterable[modules.TokenInput]
        A dataloader containing the batches on which to evaluate the model.
    device : torch.device
        The device on which the model resides.
    num_positional_embeddings : int
        Number of positional embeddings used.

    Returns
    -------
    loss_per_primitive : torch.Tensor
        A 2-dimensional tensor, of size [n, max_entities + 2] representing the loss for
        the primitives at each position for each sample in the dataloader.
    metrics : Dic[str, torch.Tensor]
        Dictionary of measured prediction metrics
    """
    if num_positional_embeddings is None:
        num_positional_embeddings = model.hparams.model.num_positional_embeddings

    all_losses = []

    metrics = {
        'kappa': CohenKappa(num_outcomes=model._kappa_metric.num_outcomes, compute_on_step=False),
        'accuracy': torchmetrics.Accuracy(num_classes=model._kappa_metric.num_outcomes, compute_on_step=False)
    }

    metrics = {
        k: v.to(device=device) for k, v in metrics.items()
    }

    for batch in dataloader:
        with torch.no_grad():
            batch_device = model.transfer_batch_to_device(batch, device)
            output: torch.Tensor = model(batch_device).to(dtype=torch.float32)

            target = batch_device['val'][..., 1:]

            loss = F.cross_entropy(
                output.swapaxes(-1, -2)[..., :-1], target,
                ignore_index=Token.Pad,
                reduction='none')

            loss_per_primitive = scatter_sum(
                loss, batch_device['pos'][..., 1:], dim_size=num_positional_embeddings + 3)

            preds = torch.distributions.Categorical(logits=output[:, :-1]).sample()
            relevant_idxs = (target != Token.Pad)

            for metric in metrics.values():
                metric(preds[relevant_idxs], target[relevant_idxs])

            # ignore first position: padding
            loss_per_primitive = loss_per_primitive[..., 2:]
            all_losses.append(loss_per_primitive.cpu().detach())

    all_losses = torch.cat(all_losses).numpy()

    metrics = {
        k: v.compute().detach().cpu().numpy() for k, v in metrics.items()
    }

    return all_losses, metrics


def plot_per_primitive_loss(results, **kwargs):
    import matplotlib.pyplot as plt
    results_bits = results / np.log(2)
    results_bits = np.ma.array(results_bits, mask=results==0)

    fig, ax = plt.subplots(**kwargs)

    ax.boxplot(
        [results_bits[:, i].compressed() for i in range(results_bits.shape[1])],
        showfliers=False)

    ax.set_xlabel('Primitive position')
    ax.set_ylabel('Negative log-likelihood (bits)')

    return fig


def compute_loss_metrics(results):
    results_bits = results / np.log(2)
    results_bits = np.ma.array(results_bits, mask=results==0)

    return {
        'bits_per_primitive': results_bits.mean(),
        'bits_per_primitive_std': results_bits.std(),
        'bits_per_sketch': results_bits.sum(axis=-1).mean(),
        'bits_per_sketch_std': results_bits.sum(axis=-1).std()
    }


@hydra.main(config_name='conf', config_path=None)
def main(config: RawPrimitiveEvaluationConfig):
    logger = logging.getLogger(__name__)

    device = torch.device('cuda')

    model: primitives_models.RawPrimitiveModule = primitives_models.RawPrimitiveModule.load_from_checkpoint(
        hydra.utils.to_absolute_path(config.checkpoint_path),
        map_location='cpu')

    model = model.eval()
    model = model.to(device=device, dtype=torch.float16)

    if config.shuffle_primitives is not None:
        model.hparams.data.permute_entities = config.shuffle_primitives

    datamodule = primitives_data.PrimitiveDataModule(
        model.hparams.data, batch_size=config.batch_size)
    datamodule.setup()

    dataloader = datamodule.test_dataloader()
    if isinstance(dataloader, list):
        if config.shuffle_primitives:
            dataloader = dataloader[-1]
        else:
            dataloader = dataloader[0]

    dataloader = tqdm.tqdm(dataloader, smoothing=0.01)
    all_losses, metrics = evaluate_model(model, dataloader, device)

    logger.info(f'Obtained classification metrics: {metrics}')
    loss_metrics = compute_loss_metrics(all_losses)
    logger.info(f'Obtained loss summary: {loss_metrics}')

    output_path = os.path.abspath('loss_per_primitive.npy')
    logger.info(f'Saving computed losses at {output_path}')
    np.save(output_path, all_losses, allow_pickle=False)

    with open('metrics.pkl', 'wb') as f:
        pickle.dump(metrics, f, protocol=4)

    fig = plot_per_primitive_loss(all_losses, figsize=(8, 4))
    fig.savefig('loss_per_primitive_position.pdf')
    fig.savefig('loss_per_primitive_position.png')



if __name__ == '__main__':
    from hydra.core.config_store import ConfigStore
    cs = ConfigStore()
    cs.store(name='conf', node=RawPrimitiveEvaluationConfig)
    main()
