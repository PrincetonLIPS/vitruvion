
import copy
import dataclasses

from typing import Optional

import hydra
import omegaconf
import pytorch_lightning

from img2cad import constraint_models, constraint_data


@dataclasses.dataclass
class ConstraintEvaluationConfig:
    checkpoint_path: str = omegaconf.MISSING
    sequence_path: Optional[str] = None


@hydra.main(config_name='conf')
def main(config: ConstraintEvaluationConfig):
    model: constraint_models.ConstraintModule = constraint_models.ConstraintModule.load_from_checkpoint(
        hydra.utils.to_absolute_path(config.checkpoint_path), map_location='cpu')

    data_config = copy.deepcopy(model.hparams.data)
    data_config.primitive_noise.enabled = True

    data_config_no_noise = copy.deepcopy(model.hparams.data)
    data_config_no_noise.primitive_noise.enabled = False

    if config.sequence_path is not None:
        data_config.sequence_path = hydra.utils.to_absolute_path(config.sequence_path)
        data_config_no_noise.sequence_path = hydra.utils.to_absolute_path(config.sequence_path)

    datamodule_noise = constraint_data.ConstraintDataModule(
        data_config, 1024)

    datamodule_no_noise = constraint_data.ConstraintDataModule(
        data_config_no_noise, 1024)

    trainer = pytorch_lightning.Trainer(
        gpus=1,
        precision=16)

    trainer.test(model, datamodule=datamodule_no_noise)
    trainer.test(model, datamodule=datamodule_noise)


if __name__ == '__main__':
    from hydra.core.config_store import ConfigStore
    cs = ConfigStore()
    cs.store('conf', node=ConstraintEvaluationConfig)
    main()

