import hydra
from hydra.core.config_store import ConfigStore

from img2cad import constraint_models, constraint_data, lightning

@hydra.main(config_name='constraints', config_path='conf')
def main(config: constraint_models.ConstraintModelTrainingConfig):
    config.data.sequence_path = hydra.utils.to_absolute_path(config.data.sequence_path)

    data = constraint_data.ConstraintDataModule(config.data, config.batch_size // config.num_gpus, config.num_data_workers // config.num_gpus)

    data.prepare_data()
    data.setup()

    config.data.dataset_size = data.train_dataset_size
    model = constraint_models.ConstraintModule(config)
    trainer = lightning.make_trainer(config)

    trainer.fit(model, datamodule=data)

if __name__ == '__main__':
    cs = ConfigStore()
    cs.store(name='constraints', node=constraint_models.ConstraintModelTrainingConfig)
    main()
