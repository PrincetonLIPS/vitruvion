"""Utilities to interact with pytorch-lightning.
"""

import dataclasses
import logging
import os
from typing import Any, Dict, List, Optional, Union

import pytorch_lightning
import pytorch_lightning.callbacks
import pytorch_lightning.plugins

import torch
import torch.distributed
from pytorch_lightning.plugins.environments.cluster_environment import ClusterEnvironment


log = logging.getLogger(__name__)

@dataclasses.dataclass
class TrainingConfiguration:
    """Base configuration for training.

    Attributes
    ----------
    max_epochs : int
        Maximum number of epochs to train for.
    num_gpus : int
        Number of GPUs to use.
    mixed_precision : bool
        If `True`, uses 16-bit training.
    ligthning
        Additional arguments to pass when creating the pytorch-lightning trainer.
    """
    max_epochs: int = 30
    num_gpus: int = 1
    mixed_precision: bool = False
    lightning: Dict[str, Any] = dataclasses.field(default_factory=dict)


def make_trainer(config: TrainingConfiguration, **kwargs) -> pytorch_lightning.Trainer:
    """Creates a new `pytorch_lightning.Trainer` according to the given configuration.

    Parameters
    ----------
    config : TrainingConfiguration
        Configuration for the trainer to create.
    **kwargs
        Additional keyword arguments which are passed to the `pytorch_lightning.Trainer` constructor.
    """
    callbacks = [
        pytorch_lightning.callbacks.ModelCheckpoint(
            save_top_k=-1,
            filename='epoch_{epoch}',
            auto_insert_metric_name=False),
        pytorch_lightning.callbacks.ModelCheckpoint(
            monitor='validation/loss',
            filename='epoch_{epoch}_vloss_{validation/loss:.3f}',
            auto_insert_metric_name=False,
            save_top_k=1),
        pytorch_lightning.callbacks.LearningRateMonitor(),
    ]

    if config.num_gpus > 0:
        callbacks.append(pytorch_lightning.callbacks.GPUStatsMonitor())

    trainer_kwargs = {
        **config.lightning
    }


    trainer_kwargs['gpus'] = config.num_gpus
    trainer_kwargs['max_epochs'] = config.max_epochs

    if config.num_gpus > 1:
        store_path = os.path.abspath('./torch_distributed_init.store')
        log.info(f'Using file-based distributed initialization at {store_path}')
        trainer_kwargs['accelerator'] = 'ddp'
        trainer_kwargs['plugins'] = DDPPlugin(init_method='file://' + store_path, find_unused_parameters=False)

    if hasattr(config, 'optim') and hasattr(config.optim, 'gradient_clip_norm'):
        if config.optim.gradient_clip_norm is not None:
            trainer_kwargs['gradient_clip_val'] = config.optim.gradient_clip_norm

    if config.mixed_precision:
        if config.num_gpus == 0:
            logging.getLogger(__name__).warn('Requested 16-bit precision but no GPUs. 16-bit precision training is not available on CPU, it has been disabled for now.')
        else:
            trainer_kwargs['precision'] = 16

    trainer_kwargs.update(kwargs)

    trainer = pytorch_lightning.Trainer(
        callbacks=callbacks,
        **trainer_kwargs)

    return trainer


class DDPPlugin(pytorch_lightning.plugins.DDPPlugin):
    """Custom DDP plugin which allows for the specification of the torch distributed initialization method.
    """
    def __init__(
        self,
        init_method: str = "env://",
        parallel_devices: Optional[List[torch.device]] = None,
        num_nodes: int = 1,
        cluster_environment: ClusterEnvironment = None,
        sync_batchnorm: bool = False,
        ddp_comm_state: Optional[object] = None,
        ddp_comm_hook: Optional[callable] = None,
        ddp_comm_wrapper: Optional[callable] = None,
        **kwargs: Union[Any, Dict[str, Any]],
    ) -> None:
        super().__init__(parallel_devices, num_nodes, cluster_environment, sync_batchnorm, ddp_comm_state, ddp_comm_hook, ddp_comm_wrapper, **kwargs)
        self.init_method = init_method

    def init_ddp_connection(self, global_rank: Optional[int]=None, world_size: Optional[int]=None) -> None:
        global_rank = global_rank if global_rank is not None else self.cluster_environment.global_rank()
        world_size = world_size if world_size is not None else self.cluster_environment.world_size()
        if not torch.distributed.is_initialized():
            log.info(f"initializing ddp: GLOBAL_RANK: {global_rank}, MEMBER: {global_rank + 1}/{world_size}")
            torch.distributed.init_process_group(
                self.torch_distributed_backend,
                init_method=self.init_method,
                rank=global_rank, world_size=world_size)
