
import copy
import dataclasses
import json

from typing import Optional

import hydra
import omegaconf
import pytorch_lightning

from img2cad import graph_models, graph_data


@dataclasses.dataclass
class GraphEvaluationConfig:
    checkpoint_path: str = omegaconf.MISSING
    sequence_path: Optional[str] = None


@hydra.main(config_name='conf', config_path=None)
def main(config: GraphEvaluationConfig):
    model: graph_models.SgAutoconstraintModel = graph_models.SketchgraphsModel.load_from_checkpoint(
        hydra.utils.to_absolute_path(config.checkpoint_path),
        map_location='cpu')

    data_config = copy.deepcopy(model.hparams.data)

    if config.sequence_path is not None:
        data_config.sequence_path = config.sequence_path

    datamodule = graph_data.SketchgraphsVitruvionDatamodule(data_config, batch_size=1024)

    trainer = pytorch_lightning.Trainer(
        gpus=1,
        precision=16)

    result, = trainer.test(model, datamodule=datamodule)
    info = {
        'num_targets': len(datamodule._dataset_test_full),
        'num_sketches': len(datamodule._dataset_test)
    }

    print(info)
    result['info'] = info

    with open('result.json', 'w') as f:
        json.dump(result, f, indent=2)


if __name__ == '__main__':
    from hydra.core.config_store import ConfigStore
    cs = ConfigStore()
    cs.store('conf', node=GraphEvaluationConfig)
    main()

