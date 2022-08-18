"""This module utilizes Optuna for hyperparameter optimization with 
the image to primitive model. 

Usage:

.. code-block:: bash

  python -m img2cad.hparam_optimization data.sequence_path=/path/to/sequence/file data.image_data_folder=/home/nrichardson/render_shards/
"""
from datetime import datetime 
from functools import partial 
import logging 
import sys

import hydra 
from hydra.core.config_store import ConfigStore
import optuna 
from optuna.trial import Trial 
from optuna.integration import PyTorchLightningPruningCallback 
import pytorch_lightning
import pytorch_lightning.callbacks

from img2cad import primitives_data, primitives_models

def objective(trial: Trial, config: primitives_models.ImagePrimitiveTrainingConfig) -> float: 
    config.model.num_positional_embeddings = trial.suggest_int("num_positional_embeddings", 12, 20)
    config.model.hidden_size = trial.suggest_int("hidden_size", 128, 512)
    config.model.num_heads = trial.suggest_int("num_heads", 2, 16)
    config.model.embedding_dimension = trial.suggest_int("embedding_dimension_mul", 4, 16) * config.model.num_heads
    config.model.num_layers = trial.suggest_int("num_layers", 2, 16) 

    config.data.sequence_path = hydra.utils.to_absolute_path(config.data.sequence_path)
    config.data.image_data_folder = hydra.utils.to_absolute_path(config.data.image_data_folder)

    data = primitives_data.ImagePrimitiveDataModule(config.data, config.batch_size // config.num_gpus, config.num_data_workers // config.num_gpus)

    data.prepare_data()
    data.setup()

    config.data.dataset_size = data.train_dataset_size

    model = primitives_models.ImagePrimitiveModule(config)

    callbacks = [
        pytorch_lightning.callbacks.ModelCheckpoint(period=1),
        pytorch_lightning.callbacks.LearningRateMonitor(),
        pytorch_lightning.callbacks.GPUStatsMonitor(), 
        PyTorchLightningPruningCallback(trial, monitor="loss")
    ]

    trainer_kwargs = {
        **config.lightning
    }

    trainer_kwargs['gpus'] = config.num_gpus
    trainer_kwargs['max_epochs'] = config.max_epochs

    if config.num_gpus > 1:
        trainer_kwargs['accelerator'] = 'ddp'

    if config.optim.gradient_clip_norm is not None:
        trainer_kwargs['gradient_clip_val'] = config.optim.gradient_clip_norm

    if config.mixed_precision:
        trainer_kwargs['precision'] = 16

    trainer = pytorch_lightning.Trainer(
        callbacks=callbacks,
        **trainer_kwargs)

    trainer.fit(model, datamodule=data)

@hydra.main(config_name='config')
def main(config: primitives_models.ImagePrimitiveTrainingConfig):
    pruner: optuna.pruners.BasePruner = optuna.pruners.MedianPruner()

    optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))
    study_name = datetime.now().strftime("study_%Y%m%d_%H%M") 
    storage_name = "sqlite:///{}.db".format(study_name)
    study = optuna.create_study(direction="minimize", pruner=pruner, study_name=study_name, storage=storage_name)
    study.optimize(partial(objective, config=config), n_trials=20, timeout=12 * (60 * 60))


if __name__=="__main__": 
    cs = ConfigStore()
    cs.store(name='config', node=primitives_models.ImagePrimitiveTrainingConfig)
    main()
