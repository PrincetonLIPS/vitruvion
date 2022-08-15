"""This script trains the sketchgraphs baseline model.
"""

import hydra

from . import lightning, graph_data, graph_models


@hydra.main(config_name='sketchgraphs', config_path='conf')
def main(config: graph_models.SketchgraphsTrainingConfig):
    dm = graph_data.SketchgraphsVitruvionDatamodule(config.data, config.batch_size, config.data.num_workers)
    model = graph_models.SketchgraphsModel(config)

    trainer = lightning.make_trainer(config)
    trainer.fit(model, datamodule=dm)

if __name__ == '__main__':
    from hydra.core.config_store import ConfigStore
    cs = ConfigStore()
    cs.store(name='base_sketchgraphs', node=graph_models.SketchgraphsTrainingConfig)
    main()
