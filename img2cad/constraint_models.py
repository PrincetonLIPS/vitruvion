"""Main pytorch-lightning models for training constraint problem.
"""

import dataclasses
from typing import Optional

import numpy as np
import pytorch_lightning
import torch
import torch.nn
import torchmetrics

from img2cad import constraint_data, modules, lightning
from sketchgraphs_models.nn.summary import CohenKappa


@dataclasses.dataclass
class ConstraintModelConfig:
    """Configuration for the main constraint model.

    Attributes
    ----------
    num_position_embeddings
        Maximum number of entities in the sketch
    embedding_dimension
        Dimension of embeddings used
    hidden_size
        Dimension of hidden layers
    num_heads
        Number of heads used in multi-head attention
    num_layers
        Number of transformer layers
    """
    num_positional_embeddings: int = 16
    embedding_dimension: int = 128
    hidden_size: int = 256
    num_heads: int = 4
    num_layers: int = 8

@dataclasses.dataclass
class OptimConfig:
    learning_rate: float = 1e-4
    gradient_clip_norm: Optional[float] = 1.0


@dataclasses.dataclass
class ConstraintModelTrainingConfig(lightning.TrainingConfiguration):
    data: constraint_data.ConstraintDataConfig = constraint_data.ConstraintDataConfig()
    model: ConstraintModelConfig = ConstraintModelConfig()
    optim: OptimConfig = OptimConfig()
    batch_size: int = 256
    num_data_workers: int = 8


class ConstraintModel(torch.nn.Module):
    """Autoregressive generative model of constraints conditioned on primitives.

    Primitives are first embedded via a non-masked PrimitiveModel.
    Constraints are sampled via a masked decoder, using the learned primitive
    embeddings to form dynamic constraint embeddings.
    """
    def __init__(self,
                 num_bins,
                 max_entities,
                 embed_dim,
                 fc_size,
                 num_heads,
                 num_layers,
                 dropout=0):
        super(ConstraintModel, self).__init__()
        # Primitive model (for dynamic embeddings)
        self.prim_model = modules.PrimitiveModel(num_bins, max_entities, embed_dim,
            fc_size, num_heads, num_layers, dropout, use_mask=False,
            linear_decode=False)
        pad_tok = constraint_data.Token.Pad
        # Value embeddings (only fixed ones)
        num_val_embeddings = len(constraint_data.Token)
        self.val_embed = torch.nn.Embedding(num_val_embeddings, embed_dim,
            padding_idx=pad_tok)
        # Coordinate embeddings
        num_coord_embeddings = 2 + len(
            constraint_data.CONSTRAINT_COORD_TOKENS)
        self.coord_embed = torch.nn.Embedding(num_coord_embeddings, embed_dim,
            padding_idx=pad_tok)
        # Position embeddings
        num_pos_embeddings = 3 + (4 * max_entities)  # see make_sequence_dataset
        self.pos_embed = torch.nn.Embedding(num_pos_embeddings, embed_dim,
            padding_idx=pad_tok)  # TODO: dry-ify overlapping logic w/ PrimModel
        # Transformer decoder
        decoder_layers = torch.nn.TransformerDecoderLayer(embed_dim, num_heads, fc_size,
            dropout)
        self.trans_decoder = torch.nn.TransformerDecoder(decoder_layers, num_layers)

    def _embed_tokens(self, src):
        # Embed primitives
        prim_embeddings = self.prim_model(src)
        # Prepend fixed val embeddings (constraint types)
        batch_size = src['c_val'].shape[0]
        fixed_tokens = np.tile(range(len(constraint_data.Token)), (batch_size, 1))
        fixed_tokens = torch.tensor(fixed_tokens).to(src['c_val'].device)
        fixed_val_embeddings = self.val_embed(fixed_tokens)
        prim_embeddings = torch.cat([fixed_val_embeddings, prim_embeddings], 1)
        val_tokens = torch.unsqueeze(src['c_val'], 2).expand(
            -1, -1, prim_embeddings.shape[2])
        # Embed constraint tokens
        val_embeddings = torch.gather(prim_embeddings, 1, val_tokens)
        coord_embeddings = self.coord_embed(src['c_coord'])
        pos_embeddings = self.pos_embed(src['c_pos'])
        embeddings = val_embeddings + coord_embeddings + pos_embeddings
        return embeddings, prim_embeddings

    def _feed_transformer(self, c_embeddings, p_embeddings):
        # Transpose to match transformer dimensions
        c_embeddings = torch.transpose(c_embeddings, 0, 1)
        p_embeddings = torch.transpose(p_embeddings, 0, 1)
        # Pass to transformer
        tgt_mask = modules.generate_square_subsequent_mask(c_embeddings)
        output = self.trans_decoder(c_embeddings, p_embeddings, tgt_mask)
        return torch.transpose(output, 0, 1)

    def forward(self, src):
        # Embed
        c_embeddings, p_embeddings = self._embed_tokens(src)
        # Pass to transformer
        output = self._feed_transformer(c_embeddings, p_embeddings)
        # Pointer dot-product
        p_embeddings = torch.transpose(p_embeddings, 1, 2)
        output = torch.matmul(output, p_embeddings)
        return output



class ConstraintModule(pytorch_lightning.LightningModule):
    """Main constraint model implementation.
    """
    hparams: ConstraintModelTrainingConfig

    def __init__(self, config: ConstraintModelTrainingConfig):
        super().__init__()

        self.save_hyperparameters(config)
        self.model = ConstraintModel(
            self.hparams.data.num_position_bins,
            self.hparams.model.num_positional_embeddings,
            self.hparams.model.embedding_dimension,
            self.hparams.model.hidden_size,
            self.hparams.model.num_heads,
            self.hparams.model.num_layers)

        self.criterion = torch.nn.CrossEntropyLoss(ignore_index=constraint_data.Token.Pad)

        num_classes = len(constraint_data.Token) + config.data.max_token_length
        self._kappa_metric = CohenKappa(len(constraint_data.Token) + config.data.max_token_length)
        self._accuracy_metric = torchmetrics.Accuracy(num_classes=num_classes)

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

    def forward(self, batch: modules.TokenInput) -> torch.Tensor:
        return self.model(batch)

    def _compute_loss(self, output: torch.Tensor, batch: modules.TokenInput) -> torch.Tensor:
        output = output[:, :-1].reshape(-1, output.shape[-1])
        target = batch['c_val'][:, 1:].reshape(-1)

        preds = torch.distributions.Categorical(logits=output).sample()

        relevant_idxs = (target != constraint_data.Token.Pad)
        self._kappa_metric(preds[relevant_idxs], target[relevant_idxs])
        self._accuracy_metric(preds[relevant_idxs], target[relevant_idxs])

        return self.criterion(output, target)

    def training_step(self, batch: modules.TokenInput, batch_idx):
        output = self.forward(batch)
        loss = self._compute_loss(output, batch)

        self.log('loss', loss)
        self.log('kappa', self._kappa_metric, prog_bar=True)

        return loss

    def validation_step(self, batch: modules.TokenInput, batch_idx):
        output = self.forward(batch)
        loss = self._compute_loss(output, batch)

        self.log('validation/loss', loss)
        self.log('validation/kappa', self._kappa_metric)

    def test_step(self, batch: modules.TokenInput, batch_idx):
        output = self.forward(batch)
        loss = self._compute_loss(output, batch)

        self.log('test/loss', loss)
        self.log('test/kappa', self._kappa_metric)
        self.log('test/accuracy', self._accuracy_metric)
