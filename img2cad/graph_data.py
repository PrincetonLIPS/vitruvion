"""Data compatibility layers for reading Vitruvion data into Sketchgraphs models.

This functionality is used for comparison between the Vitruvion and Sketchgraphs models.

"""

import dataclasses
import functools
from typing import Any, Dict, List, Optional, Sequence, Type, Union

import numpy as np
import torch
import torch.utils.data
import pytorch_lightning

from sketchgraphs import data as datalib
from sketchgraphs.data import flat_array
from sketchgraphs.pipeline import graph_model
from sketchgraphs.pipeline.graph_model.target import TargetType

import sketchgraphs_models.autoconstraint.dataset
import sketchgraphs_models.graph.dataset
import sketchgraphs_models.graph.model

from . import data_utils as i2c_utils
from .primitives_data import PrimitiveDataConfig, split_dataset
from . import constraint_data

SeqOp = Union[datalib.NodeOp, datalib.EdgeOp]


NODE_TYPES_PREDICTED = [datalib.EntityType.Arc, datalib.EntityType.Circle, datalib.EntityType.Line, datalib.EntityType.Point, datalib.EntityType.Stop]
NODE_TYPES = NODE_TYPES_PREDICTED + [datalib.EntityType.External] + list(datalib.SubnodeType)

EDGE_TYPES_PREDICTED = [
    datalib.ConstraintType.Coincident,
    datalib.ConstraintType.Concentric,
    datalib.ConstraintType.Equal,
    datalib.ConstraintType.Fix,
    datalib.ConstraintType.Horizontal,
    datalib.ConstraintType.Midpoint,
    datalib.ConstraintType.Normal,
    datalib.ConstraintType.Offset,
    datalib.ConstraintType.Parallel,
    datalib.ConstraintType.Perpendicular,
    datalib.ConstraintType.Quadrant,
    datalib.ConstraintType.Tangent,
    datalib.ConstraintType.Vertical,
]

EDGE_TYPES = EDGE_TYPES_PREDICTED + [datalib.ConstraintType.Subnode]

EDGE_IDX_MAP = {t: i for i, t in enumerate(EDGE_TYPES)}
NODE_IDX_MAP = {t: i for i, t in enumerate(NODE_TYPES)}


TARGET_TYPE_TO_LABEL_AND_ENTITY = {
    TargetType.NodeArc: (datalib.EntityType.Arc, datalib.Arc),
    TargetType.NodeCircle: (datalib.EntityType.Circle, datalib.Circle),
    TargetType.NodeLine: (datalib.EntityType.Line, datalib.Line),
    TargetType.NodePoint: (datalib.EntityType.Point, datalib.Point),
}


class VitruvionEntityMapping:
    """Class implementing a compatibility layer to featurize Vitruvion features
    for use in Sketchgraphs models (GNNs). This class implements an interface
    which is compatible with `NodeEntityMapping`.
    """
    num_bins: int

    def __init__(self, num_bins: int = 64):
        self.num_bins = num_bins

    def _numerical_features(self, ops: Sequence[datalib.NodeOp], EntityType: Type[datalib.Entity]):
        features = []

        for op in ops:
            ent = EntityType("", **op.parameters)
            ent_params = i2c_utils.parameterize_entity(ent)

            # Clip to range of allowed parameters
            ent_params = np.clip(ent_params, -0.5, 0.5)
            ent_params_quantized = i2c_utils.quantize_params(ent_params, self.num_bins)

            ent_features = np.concatenate(([ent.isConstruction], ent_params_quantized))
            features.append(ent_features)

        if len(features) == 0:
            # Handle case where no entities of given type exist.
            features = np.zeros((0, i2c_utils.NUM_PARAMS[EntityType] + 1), dtype=np.int64)
        else:
            features = np.stack(features, axis=0).astype(np.int64)

        return features

    def numerical_features(self, ops: Sequence[datalib.NodeOp], target: TargetType) -> np.ndarray:
        label, EntityType = TARGET_TYPE_TO_LABEL_AND_ENTITY[target]
        ops = [op for op in ops if op.label == label]
        return self._numerical_features(ops, EntityType)


    def sparse_features_for_target(self, node_ops: Sequence[datalib.NodeOp], target: TargetType):
        """Produces sparse features for given target type.

        This function produces a sparse feature batch for the given sequence of ops
        and target type. The sparse features are emitted for each node in `node_ops` which
        matches the specified target type.
        """
        index = []
        ops = []

        label, _ = TARGET_TYPE_TO_LABEL_AND_ENTITY[target]

        for i, e in enumerate(node_ops):
            if e.label != label:
                continue
            index.append(i)
            ops.append(e)

        index = np.array(index, dtype=np.int64)
        features = self.numerical_features(ops, target)

        return graph_model.SparseFeatureBatch(index, features)

    def all_sparse_features(self, node_ops: Sequence[datalib.NodeOp]):
        """Returns a dictionary of sparse features corresponding to the operations in
        the given sequence, grouped by operation type.
        """
        return {
            target: self.sparse_features_for_target(node_ops, target)
            for target in TargetType.numerical_node_types()
        }


def make_node_feature_dimensions(num_bins: int) -> Dict[TargetType, Dict[int, int]]:
    def _list_to_dict(s):
        return dict(zip(range(len(s)), s))
    node_feature_dimensions = {
        t: _list_to_dict([2] + [num_bins] * i2c_utils.NUM_PARAMS[TARGET_TYPE_TO_LABEL_AND_ENTITY[t][1]])
        for t in TargetType.numerical_node_types()
    }
    return node_feature_dimensions


@dataclasses.dataclass
class SketchgraphsDataConfig(PrimitiveDataConfig):
    """Configuration for loading Vitruvion data as Sketchgraphs data.
    """
    num_workers: int = 16


class SketchgraphsVitruvionDatamodule(pytorch_lightning.LightningDataModule):
    """This module implements the main data loading logic for running Sketchgraph models on the
    Vitruvion dataset.
    """
    def __init__(self, config: SketchgraphsDataConfig, batch_size: int, num_workers: int=8):
        super().__init__()
        self.config = config
        self.num_workers = num_workers
        self.batch_size = batch_size
        self._entity_mapping = VitruvionEntityMapping(self.config.num_position_bins)

    def setup(self, stage=None):
        self.dataset = sketchgraphs_models.graph.dataset.GraphDataset(
            flat_array.load_dictionary_flat(self.config.sequence_path)['sequences'],
            self._entity_mapping,
            node_idx_map=NODE_IDX_MAP,
            edge_idx_map=EDGE_IDX_MAP)

        ds_train, ds_val, ds_test = split_dataset(
            self.dataset,
            self.config.validation_fraction,
            self.config.test_fraction)

        self._dataset_train = ds_train
        self._dataset_val = ds_val
        self._dataset_test = ds_test

        self._dataset_test_full = None

    def _make_dataloader(self, ds, shuffle: bool=True):
        return torch.utils.data.DataLoader(
            ds,
            self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            collate_fn=functools.partial(
                sketchgraphs_models.graph.dataset.collate,
                entity_feature_mapping=self._entity_mapping,
                edge_feature_mapping=None,
                node_idx_map=NODE_IDX_MAP,
                edge_idx_map=EDGE_IDX_MAP),
            persistent_workers=self.num_workers > 0)

    def train_dataloader(self):
        return self._make_dataloader(self._dataset_train, shuffle=True)

    def val_dataloader(self):
        return self._make_dataloader(self._dataset_val, shuffle=False)

    def test_dataloader(self):
        if self._dataset_test_full is None:
            self._dataset_test_full = sketchgraphs_models.graph.dataset.FullTargetsGraphDataset(
                torch.utils.data.Subset(self.dataset.sequences, self._dataset_test.indices),
                self.dataset.node_feature_mapping,
                self.dataset.edge_feature_mapping,
                self.dataset.node_idx_map,
                self.dataset.edge_idx_map)
        return self._make_dataloader(self._dataset_test, shuffle=False)


class MapDataset(torch.utils.data.Dataset):
    """Utility class for creating a dataset which applies the given function to each element."""
    def __init__(self, dataset: Sequence[Any], map_fn):
        self.dataset = dataset
        self.map_fn = map_fn

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        return self.map_fn(self.dataset[index])


def apply_primitive_noise(seq: Sequence[SeqOp]) -> Sequence[SeqOp]:
    sketch = datalib.sketch_from_sequence(seq)
    sketch = constraint_data.apply_primitive_noise(sketch)
    return datalib.sketch_to_sequence(sketch)


class AutoconstrainVitruvionDatamodule(pytorch_lightning.LightningDataModule):
    """Datamodule for loading Vitruvion data into Sketchgraphs Autoconstrain model.
    """
    _dataset_test_full: Optional[sketchgraphs_models.autoconstraint.dataset.FullTargetsAutoconstraintDataset] = None

    def __init__(self, config: SketchgraphsDataConfig, batch_size: int, num_workers: int=8, primitive_noise: bool=False):
        super().__init__()
        self.config = config
        self.batch_size = batch_size
        self.num_workers = num_workers
        self._entity_mapping = VitruvionEntityMapping(self.config.num_position_bins)

        self._dataset_test_full = None
        self.primitive_noise = primitive_noise

    def setup(self, stage=None):
        sequences = flat_array.load_dictionary_flat(self.config.sequence_path)['sequences']

        if self.primitive_noise:
            sequences = MapDataset(sequences, apply_primitive_noise)

        self.dataset = sketchgraphs_models.autoconstraint.dataset.AutoconstraintDataset(
            sequences,
            node_feature_mappings=self._entity_mapping,
            node_idx_map=NODE_IDX_MAP,
            edge_idx_map=EDGE_IDX_MAP,
            no_external_constraints=True)

        ds_train, ds_val, ds_test = split_dataset(
            self.dataset,
            self.config.validation_fraction,
            self.config.test_fraction)

        self._dataset_train = ds_train
        self._dataset_val = ds_val
        self._dataset_test = ds_test
        self._dataset_test_full = None

    def _make_dataloader(self, ds, shuffle: bool=True):
        return torch.utils.data.DataLoader(
            ds,
            self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            collate_fn=sketchgraphs_models.autoconstraint.dataset.collate,
            persistent_workers=self.num_workers > 0)

    def train_dataloader(self):
        return self._make_dataloader(self._dataset_train, shuffle=True)

    def val_dataloader(self):
        return self._make_dataloader(self._dataset_val, shuffle=False)

    def test_dataloader(self):
        if self._dataset_test_full is None:
            self._dataset_test_full = sketchgraphs_models.autoconstraint.dataset.FullTargetsAutoconstraintDataset.from_dataset(
                self.dataset,
                torch.utils.data.Subset(self.dataset.sequences, self._dataset_test.indices))

        return self._make_dataloader(self._dataset_test_full, shuffle=False)
