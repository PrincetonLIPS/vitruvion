import dataclasses
import logging

import hydra
import pytorch_lightning
import torch
import tqdm

from img2cad import dataset, modules, primitives_data
from img2cad.evaluation.evaluate_raw_primitives import evaluate_model, compute_loss_metrics
from sketchgraphs_models.nn.summary import CohenKappa


@dataclasses.dataclass
class UniformEvaluationConfig:
    data: primitives_data.RawPrimitiveDataConfig = primitives_data.RawPrimitiveDataConfig()


class UniformModule(pytorch_lightning.LightningModule):
    dummy: torch.Tensor

    def __init__(self, num_outcomes: int):
        super().__init__()

        self.num_outcomes = num_outcomes
        self.register_parameter('dummy', torch.nn.Parameter(torch.zeros(1)))

        # dummy metric for compatibility
        self._kappa_metric = CohenKappa(num_outcomes)


    def forward(self, data: modules.TokenInput):
        return self.dummy.new_ones((data['coord'].shape[0], data['coord'].shape[1], self.num_outcomes))


@hydra.main(config_name='conf')
def main(config: UniformEvaluationConfig):
    logger = logging.getLogger(__name__)

    datamodule = primitives_data.PrimitiveDataModule(config.data, batch_size=4096)
    datamodule.setup()

    dataloader = tqdm.tqdm(datamodule.test_dataloader(), smoothing=0.01)

    num_outcomes = len(dataset.Token) + config.data.num_position_bins + 2
    model = UniformModule(num_outcomes)

    all_losses, metrics = evaluate_model(model, dataloader, torch.device('cpu'), num_positional_embeddings=16)
    loss_metrics = compute_loss_metrics(all_losses)

    logger.info(f'Obtained classification metrics: {metrics}')
    loss_metrics = compute_loss_metrics(all_losses)
    logger.info(f'Obtained loss summary: {loss_metrics}')


if __name__ == '__main__':
    from hydra.core.config_store import ConfigStore
    cs = ConfigStore()
    cs.store(name='conf', node=UniformEvaluationConfig)
    main()
