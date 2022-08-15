"""Sketchgraphs models adapted for Vitruvion problem.
"""

import dataclasses

import torch
import torch.utils.data
import pytorch_lightning
import omegaconf

from sketchgraphs.pipeline.graph_model.target import TargetType

from sketchgraphs_models import nn as sg_nn
import sketchgraphs_models.autoconstraint.model
import sketchgraphs_models.graph.dataset
import sketchgraphs_models.graph.model
from sketchgraphs_models.graph.model import message_passing, numerical_features, GraphModel, EdgePartnerNetwork

from . import lightning
from . import graph_data


def make_graph_model(hidden_size: int, message_passing_rounds: int, num_bins: int):
    """Construct a graph model with the given hyper-parameters for the Vitruvion problem.

    This function constructs a `GraphModel` specialized to the Vitruvion problem encoding.
    It operates with the following assumptions:
    - nodes
        - only 4 main node types (Arc, Circle, Line, Point), and subnode types
        - each node encodes label, isConstruction, and quantized position parameters
    - edges
        - only purely categorical edges are considered
        - edges encode no feature except label
    """
    node_feature_dimensions = graph_data.make_node_feature_dimensions(num_bins)
    edge_embedding = message_passing.DenseOnlyEmbedding(len(graph_data.EDGE_TYPES), hidden_size)
    node_embeddings, node_readouts = numerical_features.make_embedding_and_readout(
        hidden_size, node_feature_dimensions, numerical_features.entity_decoder_initial_input)

    node_embedding = message_passing.DenseSparsePreEmbedding(
        TargetType, node_embeddings, len(graph_data.NODE_TYPES), hidden_size)

    # Build main model core
    model_core = message_passing.GraphModelCore(
        sg_nn.MessagePassingNetwork(
            message_passing_rounds,
            torch.nn.GRUCell(hidden_size, hidden_size),
            sg_nn.ConcatenateLinear(hidden_size, hidden_size, hidden_size)),
        node_embedding,
        edge_embedding,
        message_passing.GraphPostEmbedding(hidden_size, hidden_size),
    )

    return GraphModel(
        model_core,
        entity_label=sg_nn.Sequential(
            torch.nn.Linear(hidden_size, hidden_size // 2),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size // 2, hidden_size // 2),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size // 2, len(graph_data.NODE_TYPES_PREDICTED))
        ),
        entity_feature_readout=node_readouts,
        edge_post_embedding=sg_nn.ConcatenateLinear(hidden_size, hidden_size, hidden_size),
        edge_label=sg_nn.Sequential(
            sg_nn.ConcatenateLinear(hidden_size, hidden_size, hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size, hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size, len(graph_data.EDGE_TYPES_PREDICTED))
        ),
        edge_feature_readout=None,
        edge_partner=EdgePartnerNetwork(
            torch.nn.Sequential(
                torch.nn.Linear(3 * hidden_size, hidden_size),
                torch.nn.ReLU(),
                torch.nn.Linear(hidden_size, 1))))


@dataclasses.dataclass
class SketchgraphsModelConfig:
    hidden_size: int = 384
    depth: int = 3

@dataclasses.dataclass
class OptimConfig:
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4

@dataclasses.dataclass
class SketchgraphsTrainingConfig(lightning.TrainingConfiguration):
    model: SketchgraphsModelConfig = SketchgraphsModelConfig()
    data: graph_data.SketchgraphsDataConfig = graph_data.SketchgraphsDataConfig()
    optim: OptimConfig = OptimConfig()
    batch_size: int = 2048


def _total_loss(losses):
    result = 0
    for v in losses.values():
        if v is None:
            continue
        if isinstance(v, dict):
            result += _total_loss(v)
        else:
            result += v.sum()
    return result


class SketchgraphsModel(pytorch_lightning.LightningModule):
    """Main module for training Sketchgraphs generative model on the Vitruvion problem.
    """
    hparams: SketchgraphsTrainingConfig

    def __init__(self, config: SketchgraphsTrainingConfig):
        super().__init__()

        if not omegaconf.OmegaConf.is_config(config):
            config = omegaconf.OmegaConf.structured(config)

        self.save_hyperparameters(config)
        self.model = make_graph_model(
            hidden_size=self.hparams.model.hidden_size,
            message_passing_rounds=self.hparams.model.depth,
            num_bins=self.hparams.data.num_position_bins)

        self.feature_dimensions = graph_data.make_node_feature_dimensions(self.hparams.data.num_position_bins)
        self._global_keys = [
            'edge_label', 'edge_partner', 'node_label', 'node_stop', 'subnode_stop'
        ]

    def forward(self, data):
        return self.model(data)

    def _step(self, data, prefix=''):
        readout = self(data)
        losses, accuracy, _, node_metrics = sketchgraphs_models.graph.model.compute_losses(
            readout, data, self.feature_dimensions)

        total_loss = _total_loss(losses)
        loss = total_loss / sum(data['graph_counts'])

        avg_losses = sketchgraphs_models.graph.model.compute_average_losses(losses, data['graph_counts'])
        self.log_dict({prefix + 'loss/' + k: avg_losses[k] for k in self._global_keys})
        self.log(prefix + 'loss/total', loss)
        self.log_dict({prefix + 'accuracy/' + k: accuracy[k] for k in self._global_keys})

        return loss

    def training_step(self, data, batch_idx):
        return self._step(data)

    def validation_step(self, data, batch_idx):
        loss = self._step(data, prefix='validation_')
        self.log('validation/loss', loss)
        return loss

    def test_step(self, data, batch_idx):
        return self._step(data, prefix='test_')

    def configure_optimizers(self):
        optim = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.hparams.optim.learning_rate * self.hparams.batch_size / 256,
            weight_decay=self.hparams.optim.weight_decay)

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=self.hparams.max_epochs)

        return [optim], [scheduler]


class SgAutoconstraintModel(pytorch_lightning.LightningModule):
    hparams: SketchgraphsTrainingConfig

    def __init__(self, config: SketchgraphsTrainingConfig):
        super().__init__()

        if not omegaconf.OmegaConf.is_config(config):
            config = omegaconf.OmegaConf.structured(config)

        self.save_hyperparameters(config)

        feature_dimensions = graph_data.make_node_feature_dimensions(self.hparams.data.num_position_bins)

        core = sketchgraphs_models.autoconstraint.model.BidirectionalRecurrentModelCore(
            self.hparams.model.hidden_size,
            feature_dimensions)
        self.model = sketchgraphs_models.autoconstraint.model.AutoconstraintModel(core)

    def forward(self, batch):
        return self.model(batch)

    def _step(self, batch, prefix=''):
        readout = self(batch)
        losses, accuracy = sketchgraphs_models.autoconstraint.model.compute_losses(batch, readout)
        total_loss = sum(losses.values())
        loss = total_loss / batch['graph'].node_counts.shape[0]

        with torch.no_grad():   
            average_losses = sketchgraphs_models.autoconstraint.model.compute_average_losses(batch, losses)

        self.log_dict({prefix + 'loss/' + k: v for k, v in average_losses.items()})
        self.log_dict({prefix + 'accuracy/' + k: v for k, v in accuracy.items()})

        return loss

    def training_step(self, batch, batch_idx):
        return self._step(batch)

    def validation_step(self, batch, batch_idx):
        loss = self._step(batch, 'validation_')
        self.log('validation/loss', loss)

    def test_step(self, batch, batch_idx):
        loss = self._step(batch, 'test_')
        self.log('test/loss', loss)

    def configure_optimizers(self):
        optim = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.hparams.optim.learning_rate * self.hparams.batch_size / 256,
            weight_decay=self.hparams.optim.weight_decay)

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=self.hparams.max_epochs)

        return [optim], [scheduler]
