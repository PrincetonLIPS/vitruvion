
import copy
import dataclasses
import json
import os

from typing import Optional

import hydra
import omegaconf
import pytorch_lightning

from img2cad import graph_models, graph_data


@dataclasses.dataclass
class ConstraintEvaluationConfig:
    checkpoint_path: str = omegaconf.MISSING
    sequence_path: Optional[str] = None
    primitive_noise: bool = False


@hydra.main(config_name='conf', config_path=None)
def main(config: ConstraintEvaluationConfig):
    model: graph_models.SgAutoconstraintModel = graph_models.SgAutoconstraintModel.load_from_checkpoint(
        hydra.utils.to_absolute_path(config.checkpoint_path),
        map_location='cpu')

    data_config = copy.deepcopy(model.hparams.data)
    data_config.num_workers = len(os.sched_getaffinity(0))

    if config.sequence_path is not None:
        data_config.sequence_path = config.sequence_path

    datamodule = graph_data.AutoconstrainVitruvionDatamodule(data_config, batch_size=1024, primitive_noise=config.primitive_noise)
    datamodule.setup()
    _ = datamodule.test_dataloader()

    test_targets = len(datamodule._dataset_test_full)
    test_sketches = len(datamodule._dataset_test)
    test_tokens = int(sum(sum(len(op.references) for op in seq.edge_ops) for seq in datamodule._dataset_test_full._sequence_info))

    info_dict = {
        'num_targets': test_targets,
        'num_sketches': test_sketches,
        'num_tokens': test_tokens
    }

    print(info_dict)

    trainer = pytorch_lightning.Trainer(
        gpus=1,
        precision=16)

    result, = trainer.test(model, datamodule=datamodule)
    result['info'] = info_dict
    with open('test_result.json', 'wt') as f:
        json.dump(result, f, indent=2)


if __name__ == '__main__':
    from hydra.core.config_store import ConfigStore
    cs = ConfigStore()
    cs.store('conf', node=ConstraintEvaluationConfig)
    main()

