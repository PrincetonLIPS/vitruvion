"""Main preprocessing strategies for model that encompasses the "subnode" model plus both constraint and entity parameters. 

All numerical parameters are represented in a one-hot fashion.
"""

import collections
import enum
import functools
import itertools
import json
from typing import Any, Dict, List, Tuple, Union, Sequence

import numpy as np
import torch
import torch.utils.data

from sketchgraphs.data import sketch as data_sketch, sequence as data_sequence

from sketchgraphs.pipeline.graph_model import GraphInfo, SparseFeatureBatch
from sketchgraphs.pipeline.graph_model.quantization import EdgeFeatureMapping, EntityFeatureMapping, QuantizationMap
from sketchgraphs.pipeline.graph_model.target import TargetType, NODE_IDX_MAP, EDGE_IDX_MAP, NODE_IDX_MAP_REVERSE, EDGE_IDX_MAP_REVERSE


def _gather_parameters(constr):
    """Gathers relevant non-reference parameters for the given constraint.

    Parameters
    ----------
    constr: an instance of Constraint

    Returns
    -------
    parameters: a dictionary of relevant non-reference parameters
    """
    parameters = {}
    if not constr.constraintType.is_numeric:
        return parameters

    schema = [param.parameterId for param in constr.parameters]
    ref_schema = constr.constraintType.schema_comp(schema)
    if not ref_schema:
        return parameters  # return empty parameters for unsupported schema
    else:
        for param in constr.parameters:
            param_id = param.parameterId
            if param_id in ref_schema:
                if param.type == data_sketch.ConstraintParameterType.Quantity:
                    parameters[param_id] = param.expression
                elif param.type in [data_sketch.ConstraintParameterType.Enum, data_sketch.ConstraintParameterType.Boolean]:
                    parameters[param_id] = param.value
    return parameters


def _get_ent_parameters(ent):
    """Gathers float and boolean parameters for the given entity.
    """
    param_ids = type(ent).float_ids + type(ent).bool_ids
    return {param_id: getattr(ent, param_id) for param_id in param_ids}



def sketch_to_sequence(sketch):
    """Converts the given sketch to a construction sequence."""
    return data_sequence.sketch_to_sequence(sketch)


def _is_subnode_edge(op):
    return isinstance(op, data_sequence.EdgeOp) and op.label == data_sketch.ConstraintType.Subnode


def _is_stop(node_op):
    return node_op.label == data_sketch.EntityType.Stop.name


def _edge_to_tuple(edge_op: data_sequence.EdgeOp):
    if len(edge_op.references) == 1:
        return (edge_op.references[0], edge_op.references[0])
    else:
        return (edge_op.references[0], edge_op.references[1])


def _sparse_feature_to_torch(sparse_features: Dict[TargetType, SparseFeatureBatch]) -> Dict[TargetType, SparseFeatureBatch]:
    return {
        k: v.apply(torch.as_tensor) for k, v in sparse_features.items()
    }


def graph_info_from_sequence(seq, entity_feature_mapping: EntityFeatureMapping, edge_feature_mapping: EdgeFeatureMapping,
                             node_idx_map: Dict[data_sequence.EntityType, int]=None,
                             edge_idx_map: Dict[data_sequence.ConstraintType, int]=None):
    """Creates a representation of the sequence as a `GraphInfo` object.

    If specified, this function will output the desired feature maps for entities and edges.
    If not specified, they will be set to `None` instead.

    Parameters
    ----------
    seq : List[NodeOp or EdgeOp]
        A list of operations describing the construction of the sketch.
    entity_feature_mapping : EntityFeatureMapping, optional
        If not None, the mapping scheme to be used to discretize entity features.
    edge_feature_mapping : EdgeFeatureMapping, optional
        If not None, the mapping scheme to be used to discretize edge features.

    Returns
    -------
    GraphInfo
        A GraphInfo object representing the sketch.
    """
    if node_idx_map is None:
        node_idx_map = NODE_IDX_MAP
    if edge_idx_map is None:
        edge_idx_map = EDGE_IDX_MAP

    node_ops = [op for op in seq if isinstance(op, data_sequence.NodeOp) and op.label in node_idx_map]
    edge_ops = [op for op in seq if isinstance(op, data_sequence.EdgeOp) and op.label in edge_idx_map]

    # Do not explicitly encode stop node
    if _is_stop(node_ops[-1]):
        node_ops = node_ops[:-1]

    # Categorical label encoding
    edge_labels = torch.tensor([edge_idx_map[op.label] for op in edge_ops], dtype=torch.int64)
    node_labels = torch.tensor([node_idx_map[op.label] for op in node_ops], dtype=torch.int64)

    # Compute (symmetrized) edge list
    if len(edge_ops) > 0:
        incidence = torch.tensor([_edge_to_tuple(op) for op in edge_ops], dtype=torch.int64).T.contiguous()
        incidence = torch.cat((incidence, torch.flip(incidence, [0])), dim=1)
    else:
        incidence = torch.empty([2, 0], dtype=torch.int64)

    node_features = node_labels
    edge_features = edge_labels.repeat(2)

    # Compute sparse features corresponding to edge features
    if edge_feature_mapping is not None:
        sparse_edge_features = _sparse_feature_to_torch(edge_feature_mapping.all_sparse_features(edge_ops))
    else:
        sparse_edge_features = None

    # Compute sparse features corresponding to node features
    if entity_feature_mapping is not None:
        sparse_node_features = _sparse_feature_to_torch(entity_feature_mapping.all_sparse_features(node_ops))
    else:
        sparse_node_features = None

    return GraphInfo.from_single_graph(
        incidence, node_features, edge_features, len(node_labels),
        sparse_node_features, sparse_edge_features)


def _numerical_edge_targets(targets, edge_feature_mapping: EdgeFeatureMapping):
    if edge_feature_mapping is None:
        return None

    result = collections.OrderedDict()

    for target_type in TargetType.numerical_edge_types():
        result[target_type] = torch.as_tensor(edge_feature_mapping.numerical_features(targets[target_type], target_type))

    return result

def _numerical_node_targets(targets, node_feature_mapping: EntityFeatureMapping):
    if node_feature_mapping is None:
        return None

    result = collections.OrderedDict()

    for target_type in TargetType.numerical_node_types():
        result[target_type] = torch.as_tensor(node_feature_mapping.numerical_features(targets[target_type], target_type))

    return result


def _set_graph_schema(graph, entity_feature_mapping: EntityFeatureMapping, edge_feature_mapping: EdgeFeatureMapping):
    if graph.node_features is None:
        graph.node_features = graph.incidence.new_empty([0])

    if graph.edge_features is None:
        graph.edge_features = graph.incidence.new_empty([0])

    if graph.sparse_node_features is None and entity_feature_mapping is not None:
        graph.sparse_node_features = _sparse_feature_to_torch(entity_feature_mapping.all_sparse_features([]))

    if graph.sparse_edge_features is None and edge_feature_mapping is not None:
        graph.sparse_edge_features = _sparse_feature_to_torch(edge_feature_mapping.all_sparse_features([]))


def collate(batch, entity_feature_mapping: EntityFeatureMapping = None, edge_feature_mapping: EdgeFeatureMapping = None,
            node_idx_map: Dict[data_sequence.EntityType, int]=None,
            edge_idx_map: Dict[data_sequence.ConstraintType, int]=None):
    """Collates a batch of examples into a single dictionary suitable for batched computation.

    Parameters
    ----------
    batch : List
        A list of tuples representing a batch of graphs and targets.
    entity_feature_mapping : EntityFeatureMapping, optional
        A feature mapping object to obtain entity features from instances.
    edge_feature_mapping : EdgeFeatureMapping, optional
        A feature mapping object to obtain edge features from instances.

    Returns
    -------
    dict
        A dictionary containing the required features for the model.
    """
    group_graphs = [list() for _ in range(len(TargetType))]
    group_targets = [list() for _ in range(len(TargetType))]

    graph_counts = [0 for _ in range(len(TargetType))]

    for graph, target_op in batch:
        _set_graph_schema(graph, entity_feature_mapping, edge_feature_mapping)
        target_type = TargetType.from_op(target_op)
        group_graphs[target_type].append(graph)
        group_targets[target_type].append(target_op)
        graph_counts[target_type] += 1

    graph_edge_targets = GraphInfo.merge(
        *itertools.chain.from_iterable(group_graphs[t] for t in TargetType.edge_types()))

    graph_node_targets = GraphInfo.merge(
        *itertools.chain.from_iterable(group_graphs[t] for t in TargetType.node_types()))
    graph_subnode_targets = GraphInfo.merge(*group_graphs[TargetType.Subnode])

    _set_graph_schema(graph_edge_targets, entity_feature_mapping, edge_feature_mapping)
    _set_graph_schema(graph_node_targets, entity_feature_mapping, edge_feature_mapping)
    _set_graph_schema(graph_subnode_targets, entity_feature_mapping, edge_feature_mapping)

    # Labels for node / edge type prediction
    if node_idx_map is None:
        node_idx_map = NODE_IDX_MAP
    if edge_idx_map is None:
        edge_idx_map = EDGE_IDX_MAP

    node_label = torch.tensor([node_idx_map[op.label] for node_type in TargetType.node_types() for op in group_targets[node_type]],
                              dtype=torch.int64)
    edge_label = torch.tensor([edge_idx_map[op.label] for edge_type in TargetType.edge_types() for op in group_targets[edge_type]],
                              dtype=torch.int64)

    edge_partner = torch.cat([
        torch.tensor([op.references[-1] for op in group_targets[t]], dtype=torch.int64)
        for t in TargetType.edge_types()
    ]) + graph_edge_targets.node_offsets[:-1]


    graph_info = GraphInfo.merge(graph_edge_targets, graph_node_targets, graph_subnode_targets)
    edge_numerical = _numerical_edge_targets(group_targets, edge_feature_mapping)
    node_numerical = _numerical_node_targets(group_targets, entity_feature_mapping)

    result = {
        'graph': graph_info,
        'node_label': node_label,
        'edge_label': edge_label,
        'edge_partner': edge_partner,
        'graph_counts': graph_counts
    }

    if edge_numerical is not None:
        result['edge_numerical'] = edge_numerical

    if node_numerical is not None:
        result['node_numerical'] = node_numerical

    return result

ConstructionOp = Union[data_sequence.NodeOp, data_sequence.EdgeOp]

class GraphDataset(torch.utils.data.Dataset[Tuple[GraphInfo, ConstructionOp]]):
    """Dataset for numerical constraint model.

    This dataset processes the main data format for graph models, that is, it represents
    (partial) sketches as graphs which are suitable for computation using graph neural networks.
    Additionally, it also optionally outputs sparse features for the entities or the edges.
    """

    def __init__(self, sequences: Sequence[List[ConstructionOp]], node_feature_mapping=None, edge_feature_mapping=None,
                 node_idx_map: Dict[data_sequence.EntityType, int]=None,
                 edge_idx_map: Dict[data_sequence.ConstraintType, int]=None,
                 seed=None):

        if node_idx_map is None:
            node_idx_map = NODE_IDX_MAP
        if edge_idx_map is None:
            edge_idx_map = EDGE_IDX_MAP

        self.sequences = sequences
        self.rng = np.random.RandomState(seed)
        self.edge_feature_mapping = edge_feature_mapping
        self.node_feature_mapping = node_feature_mapping

        self.node_idx_map = node_idx_map
        self.edge_idx_map = edge_idx_map

    def __len__(self):
        return len(self.sequences)


    def _op_valid(self, op: ConstructionOp) -> bool:
        if isinstance(op, data_sequence.NodeOp):
            return op.label in self.node_idx_map
        else:
            return op.label in self.edge_idx_map


    def _filter_sequence(self, seq: Sequence[ConstructionOp]) -> Sequence[ConstructionOp]:
        return [op for op in seq if self._op_valid(op)]

    def _sequence_target_indices(self, seq: Sequence[ConstructionOp]) -> List[int]:
        return [i for i, op in enumerate(seq) if i > 0 and not _is_subnode_edge(op)]

    def _get_partial_sequence_info(self, seq, step_idx: int):
        graph = graph_info_from_sequence(
            seq[:step_idx],
            entity_feature_mapping=self.node_feature_mapping,
            edge_feature_mapping=self.edge_feature_mapping,
            node_idx_map=self.node_idx_map,
            edge_idx_map=self.edge_idx_map)

        target = seq[step_idx]

        return graph, target


    def __getitem__(self, idx):
        idx = idx % len(self)  # allows using batch size larger than dataset
        seq = self.sequences[idx]
        seq = self._filter_sequence(seq)

        # Exclude first step since we always start w/ external node
        # Exclude subnode edges since they can be inferred by the subnode op.
        step_indices = self._sequence_target_indices(seq)
        step_idx = self.rng.choice(step_indices)

        return self._get_partial_sequence_info(seq, step_idx)


class FullTargetsGraphDataset(GraphDataset):
    """Dataset for graph model.

    This dataset is similar to `GraphDataset`, but exposes all targets for each graph
    as distinct indices (instead of randomly selecting some indices).
    Note that this causes samples at different indices to be correlated, so splitting
    on this dataset is not recommended.

    """
    def __init__(self, sequences: Sequence[List[ConstructionOp]], node_feature_mapping=None, edge_feature_mapping=None,
                 node_idx_map: Dict[data_sequence.EntityType, int]=None,
                 edge_idx_map: Dict[data_sequence.ConstraintType, int]=None):
        super().__init__(sequences, node_feature_mapping, edge_feature_mapping, node_idx_map, edge_idx_map)

        target_counts = np.array([len(self._sequence_target_indices(self._filter_sequence(seq))) for seq in self.sequences], dtype=np.int32)
        target_offsets = np.zeros(len(target_counts) + 1, dtype=np.int32)
        np.cumsum(target_counts, out=target_offsets[1:])

        self._target_offsets = target_offsets

    def __len__(self):
        return self._target_offsets[-1]

    def __getitem__(self, idx):
        sequence_idx = np.searchsorted(self._target_offsets, idx, side='right') - 1
        seq = self.sequences[idx]
        seq = self._filter_sequence(seq)

        step_indices = self._sequence_target_indices(seq)
        step_idx = step_indices[idx - self._target_offsets[sequence_idx]]

        return self._get_partial_sequence_info(seq, step_idx)
