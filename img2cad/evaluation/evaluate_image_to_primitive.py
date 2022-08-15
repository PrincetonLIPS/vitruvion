
import dataclasses
import functools
import logging
import os
import pickle


import hydra
import numpy as np
import omegaconf
import torch
import tqdm

from img2cad import dataset, primitives_data, primitives_models
from img2cad.evaluation.evaluate_raw_primitives import evaluate_model, compute_loss_metrics, plot_per_primitive_loss


@dataclasses.dataclass
class ImageToPrimitiveEvaluationConfig:
    checkpoint_path: str = omegaconf.MISSING
    batch_size: int = 2048


def _convert_image_dtype(batch: dataset.ImageTokenDatum, dtype):
    batch['img'] = batch['img'].to(dtype=dtype)
    return batch


@hydra.main(config_name='conf', config_path=None)
def main(config: ImageToPrimitiveEvaluationConfig):
    logger = logging.getLogger(__name__)

    device = torch.device('cuda')
    dtype = torch.float16

    model: primitives_models.ImagePrimitiveModule = primitives_models.ImagePrimitiveModule.load_from_checkpoint(
        hydra.utils.to_absolute_path(config.checkpoint_path),
        map_location='cpu')

    model = model.eval()
    model = model.to(device=device, dtype=dtype)

    datamodule = primitives_data.ImagePrimitiveDataModule(
        model.hparams.data, batch_size=config.batch_size)
    datamodule.setup()

    dataloader = datamodule.test_dataloader()
    if isinstance(dataloader, list):
        dataloader = dataloader[0]

    dataloader = tqdm.tqdm(dataloader, smoothing=0.01)
    dataloader = map(functools.partial(_convert_image_dtype, dtype=dtype), dataloader)
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
    cs.store(name='conf', node=ImageToPrimitiveEvaluationConfig)
    main()
