
import copy
import dataclasses
import logging
import os
import pickle

from PIL import Image


import hydra
import numpy as np
import omegaconf
import torch
import torch.utils.data

from img2cad import dataset, primitives_data, primitives_models
from img2cad.evaluation.evaluate_raw_primitives import evaluate_model, compute_loss_metrics, plot_per_primitive_loss


@dataclasses.dataclass
class ImageToPrimitiveEvaluationConfig:
    checkpoint_path: str = omegaconf.MISSING
    batch_size: int = 2048


class ImagePrimitiveDataset:
    def __init__(self, tokens, image_folder):
        self.tokens = tokens
        self._idx = [10, 20, 30, 50, 60]
        self.image_folder = hydra.utils.to_absolute_path(image_folder)

    def __getitem__(self, i):
        idx = self._idx[i]

        tok = copy.copy(self.tokens[idx])
        img = Image.open(os.path.join(self.image_folder, f'{idx}.png.png'))
        img_tensor = torch.from_numpy(np.asarray(img).copy())[..., 0].unsqueeze_(0).to(torch.float32).div_(255).sub_(0.5).mul_(2).to(dtype=torch.float16)
        tok['img'] = img_tensor
        return tok

    def __len__(self):
        return len(self._idx)


@hydra.main(config_name='conf')
def main(config: ImageToPrimitiveEvaluationConfig):
    logger = logging.getLogger(__name__)

    device = torch.device('cuda')
    dtype = torch.float16

    model: primitives_models.ImagePrimitiveModule = primitives_models.ImagePrimitiveModule.load_from_checkpoint(
        hydra.utils.to_absolute_path(config.checkpoint_path),
        map_location='cpu')

    model = model.eval()
    model = model.to(device=device, dtype=dtype)

    dataset = ImagePrimitiveDataset(torch.load(hydra.utils.to_absolute_path('images.pt')), 'images')
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=5)

    all_losses, metrics = evaluate_model(model, dataloader, device)

    logger.info(f'Obtained classification metrics: {metrics}')
    loss_metrics = compute_loss_metrics(all_losses)
    logger.info(f'Obtained loss summary: {loss_metrics}')

    output_path = os.path.abspath('loss_per_primitive.npy')
    logger.info(f'Saving computed losses at {output_path}')
    np.save(output_path, all_losses, allow_pickle=False)

    with open('metrics.pkl', 'wb') as f:
        pickle.dump(metrics, f, protocol=4)


if __name__ == '__main__':
    from hydra.core.config_store import ConfigStore
    cs = ConfigStore()
    cs.store(name='conf', node=ImageToPrimitiveEvaluationConfig)
    main()
