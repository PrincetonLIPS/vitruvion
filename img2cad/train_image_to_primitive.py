"""This script trains the image to primitive model.

Usage:

.. code-block:: bash

    python -m img2cad.train_image_to_primitive data.sequence_path=/path/to/sequence/file data.image_data_folder=/home/nrichardson/render_shards/

"""


import hydra
from hydra.core.config_store import ConfigStore

from img2cad import primitives_data, primitives_models, lightning


@hydra.main(config_name='image_to_primitive', config_path='conf')
def main(config: primitives_models.ImagePrimitiveTrainingConfig):
    config.data.sequence_path = hydra.utils.to_absolute_path(config.data.sequence_path)

    if config.data.image_data_folder is not None:
        config.data.image_data_folder = hydra.utils.to_absolute_path(config.data.image_data_folder)

    data = primitives_data.ImagePrimitiveDataModule(config.data, config.batch_size // config.num_gpus, config.num_data_workers // config.num_gpus)

    data.prepare_data()
    data.setup()

    config.data.dataset_size = data.train_dataset_size
    model = primitives_models.ImagePrimitiveModule(config)
    trainer = lightning.make_trainer(config)

    trainer.fit(model, datamodule=data)

if __name__ == '__main__':
    cs = ConfigStore()
    cs.store(group='model', name='convolutional', node=primitives_models.ConvolutionalImagePrimitiveModelConfig)
    cs.store(group='model', name='base_transformer', node=primitives_models.TransformerImagePrimitiveModelConfig)
    cs.store(name='base_image_to_primitive', node=primitives_models.ImagePrimitiveTrainingConfig)
    main()
