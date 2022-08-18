"""Simple script to test a trained image to primitive model.
"""

import dataclasses
import pickle

import omegaconf
import hydra
from hydra.core.config_store import ConfigStore

from img2cad import primitives_data, primitives_models, lightning


@dataclasses.dataclass
class ImagePrimitiveTestingConfig(lightning.TrainingConfiguration):
    checkpoint_path: str = omegaconf.MISSING
    data: primitives_data.ImagePrimitiveDataConfig = primitives_data.ImagePrimitiveDataConfig()


@hydra.main(config_name='config')
def main(config: ImagePrimitiveTestingConfig):
    model = primitives_models.ImagePrimitiveModule.load_from_checkpoint(hydra.utils.to_absolute_path(config.checkpoint_path), map_location='cpu')

    config.data.sequence_path = hydra.utils.to_absolute_path(config.data.sequence_path)
    config.data.image_data_folder = hydra.utils.to_absolute_path(config.data.image_data_folder)
    data = primitives_data.ImagePrimitiveDataModule(config.data, 2048, num_workers=32)

    data.prepare_data()
    data.setup()

    config.data.dataset_size = data.train_dataset_size
    trainer = lightning.make_trainer(config)
    test_result = trainer.test(model, data.val_dataloader())

    with open('results.pkl', 'wb') as f:
        pickle.dump(test_result, f)


if __name__ == '__main__':
    cs = ConfigStore()
    cs.store(name='config', node=ImagePrimitiveTestingConfig)
    main()
