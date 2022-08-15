"""This module contains the main implementations of the model classes used for training primitive models.
It must be separated from the script module `train_primitives_raw` in order for loading models to go smoothly.
"""

import dataclasses
from typing import Any, Dict, List, Optional

import hydra
import omegaconf
import pytorch_lightning

import torch
import torch.distributions
import torch.nn
import torch.optim
import torch.utils.data

from img2cad import dataset, modules, lightning
from img2cad.primitives_data import RawPrimitiveDataConfig, ImagePrimitiveDataConfig
from sketchgraphs_models.nn.summary import CohenKappa


@dataclasses.dataclass
class RawPrimitiveModelConfig:
    """Configuration for creating a `modules.PrimitiveModel`.

    Attributes
    ----------
    num_positional_embeddings
        Number of positional embeddings to learn.
    embedding_dimension
        Dimension of the embeddings to learn.
    hidden_size
        Size of main hidden layers in transformers.
    num_heads
        Number of attention heads
    num_layers
        Number of transformer layers
    """
    num_positional_embeddings: int = 16
    embedding_dimension: int = 128
    hidden_size: int = 256
    num_heads: int = 4
    num_layers: int = 8

@dataclasses.dataclass
class ImagePrimitiveModelConfig(RawPrimitiveModelConfig):
    """Configuration for creating a `modules.ImageToPrimitiveModel`."""
    _target_: str = omegaconf.MISSING
    input_size: int = 64


@dataclasses.dataclass
class TransformerImagePrimitiveModelConfig(ImagePrimitiveModelConfig):
    """Configuration for image primitive model with visual transformer recognition.

    Attributes
    ----------
    patch_size : int
        Size of the patches that form the sequence of the embedding (in pixels).
    num_recognition_layers : Optional[int]
        If not `None`, number of layers to use for recognition model. Otherwise, recognition model
        uses the same number of layers as primitives transformer.
    """
    _target_: str = 'img2cad.primitives_models.make_transformer_image_to_primitive_model'
    patch_size: int = 8
    num_recognition_layers: Optional[int] = None


@dataclasses.dataclass
class ConvolutionalImagePrimitiveModelConfig(ImagePrimitiveModelConfig):
    """Configuration for image primitive model with convolution recognition.

    Attributes
    ----------
    width_multiplier : float
        Width multiplier for Mobilenet model used. Smaller numbers lead to smaller networks.
    """
    _target_: str = 'img2cad.primitives_models.make_convolutional_image_to_primitive_model'
    width_multiplier: float = 1.0


@dataclasses.dataclass
class OptimConfig:
    learning_rate: float = 1e-4
    gradient_clip_norm: Optional[float] = 1.0



@dataclasses.dataclass
class RawPrimitiveTrainingConfig(lightning.TrainingConfiguration):
    """Configuration for training a raw primitive model.

    Attributes
    ----------
    model : RawPrimitiveModelConfig
        Configuration for the underlying model to train.
    optim : OptimConfig
        Configuration for the optimizer used.
    data : PrimitiveDataConfig
        Configuration for dataset used to train model.
    batch_size : int
        Batch size to be used when training the model
    num_gpus : int
        Number of GPUs to use when training the model.
    num_data_workers : int
        Number of data-loading workers (per GPU)
    lightning : Dict[str, Any]
        Additional parameters to be passed to the pytorch-lightning trainer.
    """
    model: RawPrimitiveModelConfig = dataclasses.field(default_factory=RawPrimitiveModelConfig)
    optim: OptimConfig = dataclasses.field(default_factory=OptimConfig)
    data: RawPrimitiveDataConfig = RawPrimitiveDataConfig()
    batch_size: int = 128
    num_data_workers: int = 32


@dataclasses.dataclass
class ImagePrimitiveTrainingConfig(lightning.TrainingConfiguration):
    defaults: List[Any] = dataclasses.field(
        default_factory=lambda: ['_self_', {'model': 'transformer'}])
    model: ImagePrimitiveModelConfig = omegaconf.MISSING
    optim: OptimConfig = OptimConfig()
    data: ImagePrimitiveDataConfig = ImagePrimitiveDataConfig()
    batch_size: int = 128
    num_data_workers: int = 32


class RawPrimitiveModule(pytorch_lightning.LightningModule):
    """Raw generative model on primitives.

    This module encapsulates training for a plain generative model on the primitives sequence.
    """
    hparams: RawPrimitiveTrainingConfig

    def __init__(self, config: RawPrimitiveTrainingConfig):
        super().__init__()
        self.save_hyperparameters(config)

        self.model = modules.PrimitiveModel(
            num_bins=config.data.num_position_bins,
            max_entities=config.model.num_positional_embeddings,
            embed_dim=config.model.embedding_dimension,
            fc_size=config.model.hidden_size,
            num_heads=config.model.num_heads,
            num_layers=config.model.num_layers)

        self.criterion = torch.nn.CrossEntropyLoss(ignore_index=dataset.Token.Pad)
        num_outcomes = len(dataset.Token) + config.data.num_position_bins
        if dataset.INCLUDE_CONSTRUCTION:
            num_outcomes += 2
        self._kappa_metric = CohenKappa(num_outcomes)

    def configure_optimizers(self):
        base_lr = self.hparams.optim.learning_rate * self.hparams.batch_size / 128
        from apex.optimizers import FusedAdam
        AdamW = FusedAdam

        optim = AdamW(
            self.model.parameters(),
            lr=base_lr,
            weight_decay=1e-5)

        if self.hparams.data.dataset_size is None:
            raise ValueError('config.data.dataset_size must be specified to create optimizers!')

        steps_per_epoch = (self.hparams.data.dataset_size + self.hparams.batch_size - 1) // self.hparams.batch_size

        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optim,
            max_lr=base_lr,
            epochs=self.hparams.max_epochs,
            steps_per_epoch=steps_per_epoch)

        scheduler_config = {
            'scheduler': scheduler,
            'interval': 'step',
            'frequency': 1
        }

        return [optim], [scheduler_config]

    def forward(self, batch: modules.TokenInput) -> torch.Tensor:
        return self.model(batch)

    def _compute_loss(self, output: torch.Tensor, batch: modules.TokenInput) -> torch.Tensor:
        output = output[:, :-1].reshape(-1, output.shape[-1])
        target = batch['val'][:, 1:].reshape(-1)

        preds = torch.distributions.Categorical(logits=output).sample()

        relevant_idxs = (target != dataset.Token.Pad)
        self._kappa_metric(preds[relevant_idxs], target[relevant_idxs])

        return self.criterion(output, target)

    def training_step(self, batch: modules.TokenInput, batch_idx):
        output = self.forward(batch)
        loss = self._compute_loss(output, batch)

        self.log('loss', loss)
        self.log('kappa', self._kappa_metric, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=None):
        output = self.forward(batch)
        loss = self._compute_loss(output, batch)

        if dataloader_idx is None or dataloader_idx == 0:
            self.log('validation/loss', loss, add_dataloader_idx=False)
            self.log('validation/kappa', self._kappa_metric, add_dataloader_idx=False)
        else:
            self.log(f'validation_{dataloader_idx}/loss', loss, add_dataloader_idx=False)
            self.log(f'validation_{dataloader_idx}/kappa', self._kappa_metric, add_dataloader_idx=False)

    def test_step(self, batch, batch_idx, dataloader_idx=None):
        output = self.forward(batch)
        loss = self._compute_loss(output, batch)

        if dataloader_idx is None or dataloader_idx == 0:
            self.log('test/loss', loss, add_dataloader_idx=False)
            self.log('test/kappa', self._kappa_metric, add_dataloader_idx=False)
        else:
            self.log(f'test_{dataloader_idx}/loss', loss, add_dataloader_idx=False)
            self.log(f'test_{dataloader_idx}/kappa', self._kappa_metric, add_dataloader_idx=False)
        output = self.forward(batch)
        loss = self._compute_loss(output, batch)


def make_convolutional_image_to_primitive_model(
    num_positional_embeddings: int, embedding_dimension: int, hidden_size: int,
    num_heads: int, num_layers: int, input_size: int, width_multiplier: float, num_position_bins: int,
    *args, **kwargs):

    return modules.ImageToPrimitiveModel(
        num_position_bins, num_positional_embeddings,
        embedding_dimension, hidden_size, num_heads, num_layers,
        input_size, width_multiplier)


def make_transformer_image_to_primitive_model(
    num_positional_embeddings: int, embedding_dimension: int, hidden_size: int,
    num_heads: int, num_layers: int, input_size: int, patch_size: int, num_position_bins: int,
    *args, **kwargs):

    num_recognition_layers = kwargs.get('num_recognition_layers')

    return modules.SeqImageToPrimitiveModel(
        num_position_bins, num_positional_embeddings, embedding_dimension,
        hidden_size, num_heads, num_layers, input_size, patch_size,
        num_recognition_layers=num_recognition_layers)


class ImagePrimitiveModule(pytorch_lightning.LightningModule):
    hparams: ImagePrimitiveTrainingConfig

    def __init__(self, config: ImagePrimitiveTrainingConfig):
        super().__init__()

        try:
            self.save_hyperparameters(config)
        except ValueError:
            self._hparams = config

        self.model = hydra.utils.instantiate(config.model, num_position_bins=config.data.num_position_bins)

        self.criterion = torch.nn.CrossEntropyLoss(ignore_index=dataset.Token.Pad)
        num_outcomes = len(dataset.Token) + config.data.num_position_bins
        if dataset.INCLUDE_CONSTRUCTION:
            num_outcomes += 2
        self._kappa_metric = CohenKappa(num_outcomes)

    def configure_optimizers(self):
        base_lr = self.hparams.optim.learning_rate * self.hparams.batch_size / 128

        optim = torch.optim.AdamW(
            self.model.parameters(),
            lr=base_lr)

        if self.hparams.data.dataset_size is None:
            raise ValueError('config.data.dataset_size must be specified to create optimizers!')

        steps_per_epoch = (self.hparams.data.dataset_size + self.hparams.batch_size - 1) // self.hparams.batch_size

        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optim,
            max_lr=base_lr,
            epochs=self.hparams.max_epochs,
            steps_per_epoch=steps_per_epoch)

        scheduler_config = {
            'scheduler': scheduler,
            'interval': 'step',
            'frequency': 1
        }

        return [optim], [scheduler_config]


    def forward(self, batch: dataset.ImageTokenDatum):
        return self.model(batch)

    def _compute_loss(self, output: torch.Tensor, batch: modules.TokenInput) -> torch.Tensor:
        output = output[:, :-1].reshape(-1, output.shape[-1])
        target = batch['val'][:, 1:].reshape(-1)

        preds = torch.distributions.Categorical(logits=output).sample()

        relevant_idxs = (target != dataset.Token.Pad)
        self._kappa_metric(preds[relevant_idxs], target[relevant_idxs])

        return self.criterion(output, target)

    def training_step(self, batch: modules.TokenInput, batch_idx):
        output = self.forward(batch)
        loss = self._compute_loss(output, batch)

        self.log('loss', loss)
        self.log('kappa', self._kappa_metric, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=None):
        output = self.forward(batch)
        loss = self._compute_loss(output, batch)

        if dataloader_idx is None or dataloader_idx == 0:
            self.log('validation/loss', loss, add_dataloader_idx=False)
            self.log('validation/kappa', self._kappa_metric, add_dataloader_idx=False)
        else:
            self.log(f'validation_{dataloader_idx}/loss', loss, add_dataloader_idx=False)
            self.log(f'validation_{dataloader_idx}/kappa', self._kappa_metric, add_dataloader_idx=False)

    def test_step(self, batch, batch_idx, dataloader_idx=None):
        output = self.forward(batch)
        loss = self._compute_loss(output, batch)

        if dataloader_idx is None or dataloader_idx == 0:
            self.log('test/loss', loss, add_dataloader_idx=False)
            self.log('test/kappa', self._kappa_metric, add_dataloader_idx=False)
        else:
            self.log(f'test_{dataloader_idx}/loss', loss, add_dataloader_idx=False)
            self.log(f'test_{dataloader_idx}/kappa', self._kappa_metric, add_dataloader_idx=False)
        output = self.forward(batch)
        loss = self._compute_loss(output, batch)
