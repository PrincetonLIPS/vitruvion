"""This script trains the sketchgraphs baseline model.
"""

import hydra

from . import lightning, graph_data, graph_models


@hydra.main(config_name='sg_constraints', config_path='conf')
def main(config: graph_models.SketchgraphsTrainingConfig):
    dm = graph_data.AutoconstrainVitruvionDatamodule(config.data, config.batch_size, config.data.num_workers)
    model = graph_models.SgAutoconstraintModel(config)

    trainer = lightning.make_trainer(config)
    trainer.fit(model, datamodule=dm)

if __name__ == '__main__':
    from hydra.core.config_store import ConfigStore
    cs = ConfigStore()
    cs.store(name='base_sketchgraphs', node=graph_models.SketchgraphsTrainingConfig)
    main()
