"""This script trains the primitive generation model.
"""


import hydra
from hydra.core.config_store import ConfigStore

from img2cad import primitives_data, primitives_models, lightning



@hydra.main(config_name='primitives_raw', config_path='conf')
def main(config: primitives_models.RawPrimitiveTrainingConfig):
    config.data.sequence_path = hydra.utils.to_absolute_path(config.data.sequence_path)

    data = primitives_data.PrimitiveDataModule(config.data, config.batch_size // config.num_gpus, config.num_data_workers // config.num_gpus)

    data.prepare_data()
    data.setup()

    config.data.dataset_size = data.train_dataset_size
    model = primitives_models.RawPrimitiveModule(config)
    trainer = lightning.make_trainer(config)

    trainer.fit(model, datamodule=data)

if __name__ == '__main__':
    cs = ConfigStore()
    cs.store(name='base_primitives_raw', node=primitives_models.RawPrimitiveTrainingConfig)
    main()
