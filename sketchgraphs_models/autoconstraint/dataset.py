"""Dataset for auto-constraint model."""

import dataclasses
import typing
from typing import Container, List, Mapping, Union, Optional, Sequence, Type

import numpy as np
import torch
import torch.utils.data

from sketchgraphs.data import sequence as datalib
from sketchgraphs.pipeline.graph_model.target import NODE_TYPES, EDGE_TYPES, EDGE_TYPES_PREDICTED, NODE_IDX_MAP, EDGE_IDX_MAP
from sketchgraphs.pipeline import graph_model as graph_utils

from sketchgraphs_models.graph.dataset import EntityFeatureMapping, EdgeFeatureMapping, _sparse_feature_to_torch

SeqOp = Union[datalib.NodeOp, datalib.EdgeOp]

def _reindex_sparse_batch(sparse_batch, pack_batch_offsets):
    return graph_utils.SparseFeatureBatch(
        pack_batch_offsets[sparse_batch.index],
        sparse_batch.value)


def collate(batch):
    # Sort batch for packing
    node_lengths = [len(x['node_features']) for x in batch]
    sorted_indices = np.argsort(node_lengths)[::-1].copy()

    batch = [batch[i] for i in sorted_indices]

    graph = graph_utils.GraphInfo.merge(*[x['graph'] for x in batch])
    edge_label = torch.tensor(
        [x['target_edge_label'] for x in batch if x['target_edge_label'] != -1], dtype=torch.int64)
    node_features = torch.nn.utils.rnn.pack_sequence([x['node_features'] for x in batch])
    batch_offsets = graph_utils.offsets_from_counts(node_features.batch_sizes)

    node_features_graph_index = torch.cat([
        i + batch_offsets[:graph.node_counts[i]] for i in range(len(batch))
    ], dim=0)

    sparse_node_features = {}

    for k in batch[0]['sparse_node_features']:
        sparse_node_features[k] = graph_utils.SparseFeatureBatch.merge(
            [_reindex_sparse_batch(x['sparse_node_features'][k], batch_offsets) for x in batch], range(len(batch)))

    last_graph_node_index = batch_offsets[graph.node_counts - 1] + torch.arange(len(graph.node_counts), dtype=torch.int64)

    partner_index_index = []
    partner_index = []

    stop_partner_index_index = []

    for i, x in enumerate(batch):
        if x['partner_index'] == -1:
            stop_partner_index_index.append(i)
            continue

        partner_index_index.append(i)
        partner_index.append(x['partner_index'] + graph.node_offsets[i])

    partner_index = graph_utils.SparseFeatureBatch(
        torch.tensor(partner_index_index, dtype=torch.int64),
        torch.tensor(partner_index, dtype=torch.int64)
    )

    stop_partner_index_index = torch.tensor(stop_partner_index_index, dtype=torch.int64)

    return {
        'graph': graph,
        'edge_label': edge_label,
        'partner_index': partner_index,
        'stop_partner_index_index': stop_partner_index_index,
        'node_features': node_features,
        'node_features_graph_index': node_features_graph_index,
        'sparse_node_features': sparse_node_features,
        'last_graph_node_index': last_graph_node_index,
        'sorted_indices': torch.as_tensor(sorted_indices)
    }


class AutoconstrainFeatures(typing.TypedDict):
    graph: graph_utils.GraphInfo
    node_features: torch.Tensor
    sparse_node_features: Mapping[str, graph_utils.SparseFeatureBatch]


def process_node_and_edge_ops(node_ops: Sequence[datalib.NodeOp], edge_ops_in_graph: Sequence[datalib.EdgeOp], num_nodes_in_graph: int,
                              node_feature_mappings: Optional[EntityFeatureMapping],
                              node_idx_map: Mapping[datalib.EntityType, int]=None,
                              edge_idx_map: Mapping[datalib.ConstraintType, int]=None) -> AutoconstrainFeatures:
    if node_idx_map is None:
        node_idx_map = NODE_IDX_MAP
    if edge_idx_map is None:
        edge_idx_map = EDGE_IDX_MAP

    all_node_labels = torch.tensor([node_idx_map[op.label] for op in node_ops], dtype=torch.int64)
    edge_labels = torch.tensor([edge_idx_map[op.label] for op in edge_ops_in_graph], dtype=torch.int64)

    if len(edge_ops_in_graph) > 0:
        incidence = torch.tensor([(op.references[0], op.references[-1]) for op in edge_ops_in_graph],
                                 dtype=torch.int64).T.contiguous()
        incidence = torch.cat((incidence, torch.flip(incidence, [0])), dim=1)
    else:
        incidence = torch.empty([2, 0], dtype=torch.int64)

    edge_features = edge_labels.repeat(2)

    if node_feature_mappings is not None:
        sparse_node_features = _sparse_feature_to_torch(node_feature_mappings.all_sparse_features(node_ops))
    else:
        sparse_node_features = None

    graph = graph_utils.GraphInfo.from_single_graph(incidence, None, edge_features, num_nodes_in_graph)

    return {
        'graph': graph,
        'node_features': all_node_labels,
        'sparse_node_features': sparse_node_features
    }


@dataclasses.dataclass
class SequenceConstraintInfo:
    """Constraint information for a sequence of operations.

    This class captures pre-computed information necessary to compute features
    for the auto-constraint model. This information may be derived from a single
    of operations representing a sketch by using the `identify_sequence_constraints` method.

    Attributes
    ----------
    node_ops : Sequence[datalib.NodeOp]
        A list of the node operations in the original sequence.
    edge_ops : Sequence[datalib.EdgeOp]
        A list of the edge operations in the original sequence.
    constraint_count_per_node : Sequence[int]
        A list of the number of (non-inferred) constraints for each node.
    inferred_constraint_count_per_node : Sequence[int]
        A list of the number of inferred constraints for each node.
    """
    node_ops: Sequence[datalib.NodeOp]
    edge_ops: Sequence[datalib.EdgeOp]
    constraint_count_per_node: Sequence[int]
    inferred_constraint_count_per_node: Sequence[int]

    @property
    def num_stop_targets(self) -> int:
        """The number of stop targets in the sequence.

        This correspond to the number of nodes in the sequence, as
        we must emit one stop target for each node.
        """
        return len(self.node_ops)

    @property
    def num_constraint_targets(self) -> int:
        """The number of constarin targets in the sequence.

        This corresponds to the total number of non-inferred constraints
        in the sequence.
        """
        return int(sum(self.constraint_count_per_node))


def identify_sequence_constraints(seq: Sequence[SeqOp], edge_types_inferred: Container[datalib.ConstraintType]=None) -> SequenceConstraintInfo:
    """This function identifies the number of constraints affecting each node in the sequence.

    The constraint operations are distinguished by whether they should be part of the prediction,
    or can be inferred automatically from the sequence prefix (e.g. in case of subnode constraints).

    Parameters
    ----------
    seq : Sequence[SeqOp]
        The sequence of operations to be processed.
    edge_types_inferred : Container[datalib.ConstraintType], optional
        A set of constraint types which are inferred from the sequence prefix.
        By default, this is to only be the `datalib.ConstraintType.Subnode` constraint.

    Returns
    -------
    node_ops : List[datalib.NodeOp]
        A list of the node operations in the input sequence.
    edge_ops : List[datalib.EdgeOp]
        A list of the edge operations in the input sequence.
    num_predicted_edge_ops_per_node : np.ndarray
        An integer array of the same length as `node_ops`
        containing the number of edges which are not inferred per node.
    num_non_predicted_edge_ops_per_node : np.ndarray
        An integer array of the same length as `node_ops`
        containing the number of edges which are inferred per node.
    """
    if edge_types_inferred is None:
        edge_types_inferred = [datalib.ConstraintType.Subnode]

    seq = list(seq)

    if not isinstance(seq[0], datalib.NodeOp):
        raise ValueError('First operation in sequence is not a NodeOp')

    if seq[-1].label != datalib.EntityType.Stop:
        seq.append(datalib.NodeOp(datalib.EntityType.Stop, {}))

    node_ops = [seq[0]]
    edge_ops: List[datalib.EdgeOp] = []

    num_predicted_edge_ops_per_node = []
    num_non_predicted_edge_ops_per_node = []

    predicted_edge_ops_for_current_node = 0
    non_predicted_edge_ops_for_current_node = 0

    for op in seq[1:]:
        if isinstance(op, datalib.NodeOp):
            num_predicted_edge_ops_per_node.append(predicted_edge_ops_for_current_node)
            num_non_predicted_edge_ops_per_node.append(non_predicted_edge_ops_for_current_node)

            predicted_edge_ops_for_current_node = 0
            non_predicted_edge_ops_for_current_node = 0

            node_ops.append(op)
        else:
            if op.label in edge_types_inferred:
                non_predicted_edge_ops_for_current_node += 1
            else:
                predicted_edge_ops_for_current_node += 1

            edge_ops.append(op)

    node_ops = node_ops[:-1]

    num_predicted_edge_ops_per_node = np.array(num_predicted_edge_ops_per_node, dtype=np.int64)
    num_non_predicted_edge_ops_per_node = np.array(num_non_predicted_edge_ops_per_node, dtype=np.int64)

    return SequenceConstraintInfo(node_ops, edge_ops, num_predicted_edge_ops_per_node, num_non_predicted_edge_ops_per_node)


def extract_stop_target_features(seq_info: SequenceConstraintInfo, target_node_idx: int):
    """Extracts features for predicting a stop target.

    This function extracts the required features from information given by an instance
    of `SequenceConstraintInfo` in order to predict a stop target.
    """
    if target_node_idx >= seq_info.num_stop_targets:
        raise IndexError('Target node index is out of bounds')

    predicted_edge_ops_offsets = np.cumsum(seq_info.constraint_count_per_node)
    non_predicted_edge_ops_offsets = np.cumsum(seq_info.inferred_constraint_count_per_node)

    num_nodes_in_graph = target_node_idx + 1
    edge_ops_in_graph = seq_info.edge_ops[:predicted_edge_ops_offsets[target_node_idx] + non_predicted_edge_ops_offsets[target_node_idx]]
    target_edge_label = None
    partner_index = -1

    return num_nodes_in_graph, edge_ops_in_graph, target_edge_label, partner_index


def extract_constraint_target_features(seq_info: SequenceConstraintInfo, constraint_idx: int):
    """Extracts features for predicting a constraint target.
    """
    predicted_edge_ops_offsets = np.cumsum(seq_info.constraint_count_per_node)
    non_predicted_edge_ops_offsets = np.cumsum(seq_info.inferred_constraint_count_per_node)

    if constraint_idx >= predicted_edge_ops_offsets[-1]:
        raise IndexError('Constraint index is out of bounds')

    target_predicted_edge_idx = constraint_idx
    target_node_idx = np.searchsorted(predicted_edge_ops_offsets, target_predicted_edge_idx, side='right')
    num_nodes_in_graph = target_node_idx + 1

    target_edge_idx = target_predicted_edge_idx + non_predicted_edge_ops_offsets[target_node_idx]
    target_edge = seq_info.edge_ops[target_edge_idx]
    edge_ops_in_graph = seq_info.edge_ops[:target_edge_idx]
    target_edge_label = target_edge.label
    partner_index = target_edge.references[-1]

    return num_nodes_in_graph, edge_ops_in_graph, target_edge_label, partner_index


class AutoconstraintFeatureAndLabels(AutoconstrainFeatures):
    target_edge_label: int
    partner_index: int


class AutoconstraintDataset(torch.utils.data.Dataset[AutoconstraintFeatureAndLabels]):
    def __init__(self,
                 sequences: Sequence[Sequence[SeqOp]],
                 node_feature_mappings,
                 seed: int=10,
                 node_idx_map: Mapping[datalib.EntityType, int]=None,
                 edge_idx_map: Mapping[datalib.ConstraintType, int]=None,
                 edge_types_inferred: Container[datalib.ConstraintType]=None,
                 no_external_constraints: bool=False):
        """Create a new dataset for autoconstraint prediction.

        Parameters
        ----------
        sequences : Sequence[Sequence[SeqOp]]
            Underlying set of sequences representing the dataset.
        node_feature_mappings : EntityMapping
            Mapping for sparse node features
        seed : int
            Random seed used to select the target for each sequence.
        node_idx_map : Mapping[datalib.EntityType, int]
            Mapping from entity type to one-hot integer encoding.
        edge_idx_map : Mapping[datalib.ConstraintType, int]
            Mapping from constraint type to one-hot integer encoding.
        no_external_constraints : bool
            If `True`, indicates that constraints referencing an external entity
            should not be included.
        """

        if node_idx_map is None:
            node_idx_map = NODE_IDX_MAP
        if edge_idx_map is None:
            edge_idx_map = EDGE_IDX_MAP
        if edge_types_inferred is None:
            edge_types_inferred = [datalib.ConstraintType.Subnode]

        self.sequences = sequences
        self.node_feature_mappings = node_feature_mappings
        self._rng = np.random.Generator(np.random.Philox(seed))
        self._node_idx_map = node_idx_map
        self._edge_idx_map = edge_idx_map
        self._edge_types_inferred = edge_types_inferred
        self._no_external_constraints = no_external_constraints

    @classmethod
    def from_dataset(cls: Type['AutoconstraintDataset'], dataset: 'AutoconstraintDataset', sequences: Sequence[Sequence[SeqOp]]):
        return cls(sequences, dataset.node_feature_mappings,
            dataset._rng.integers(0, 2**32 - 1),
            dataset._node_idx_map,
            dataset._edge_idx_map,
            dataset._edge_types_inferred,
            dataset._no_external_constraints)

    def __len__(self):
        return len(self.sequences)

    def _valid_constraint(self, constraint: datalib.EdgeOp):
        if self._no_external_constraints and (0 in constraint.references):
            return False

        return constraint.label in self._edge_idx_map or constraint.label in self._edge_types_inferred

    def _filter_edge_ops(self, seq: Sequence[SeqOp]) -> List[SeqOp]:
        return [op for op in seq if not isinstance(op, datalib.EdgeOp) or self._valid_constraint(op)]

    def _make_features(self, seq_info: SequenceConstraintInfo, feature_info):
        num_nodes_in_graph, edge_ops_in_graph, target_edge_label, partner_index = feature_info

        input_features = process_node_and_edge_ops(
            seq_info.node_ops, edge_ops_in_graph, num_nodes_in_graph, self.node_feature_mappings,
            self._node_idx_map, self._edge_idx_map)

        if target_edge_label is None:
            target_edge_label = -1
        else:
            target_edge_label = self._edge_idx_map[target_edge_label]

        return {
            **input_features,
            'target_edge_label': target_edge_label,
            'partner_index': partner_index,
        }

    def __getitem__(self, idx) -> AutoconstraintFeatureAndLabels:
        idx = idx % len(self.sequences)
        seq = self.sequences[idx]

        # Filter out constraints that we do not handle in the current dataset.
        seq = self._filter_edge_ops(seq)
        seq_info = identify_sequence_constraints(seq)

        stop_target = self._rng.uniform() < seq_info.num_stop_targets / (seq_info.num_stop_targets + seq_info.num_constraint_targets)

        if stop_target:
            target_node_idx = self._rng.integers(seq_info.num_stop_targets)
            feature_info = extract_stop_target_features(seq_info, target_node_idx)
        else:
            target_predicted_edge_idx = self._rng.integers(seq_info.num_constraint_targets)
            feature_info = extract_constraint_target_features(seq_info, target_predicted_edge_idx)

        return self._make_features(seq_info, feature_info)



class FullTargetsAutoconstraintDataset(AutoconstraintDataset):
    """This dataset materializes all targets for the autoconstraint problem, and presents them
    as a unified dataset. Note that this means that the samples within the dataset are not independent.
    """
    def __init__(self,
                 sequences: Sequence[Sequence[SeqOp]],
                 node_feature_mappings,
                 seed: int=10,
                 node_idx_map: Mapping[datalib.EntityType, int]=None,
                 edge_idx_map: Mapping[datalib.ConstraintType, int]=None,
                 edge_types_inferred: Container[datalib.ConstraintType]=None,
                 no_external_constraints: bool=False):
        """Create a new dataset for autoconstraint prediction.

        Parameters
        ----------
        sequences : Sequence[Sequence[SeqOp]]
            Underlying set of sequences representing the dataset.
        node_feature_mappings : EntityMapping
            Mapping for sparse node features
        node_idx_map : Mapping[datalib.EntityType, int]
            Mapping from entity type to one-hot integer encoding.
        edge_idx_map : Mapping[datalib.ConstraintType, int]
            Mapping from constraint type to one-hot integer encoding.
        """
        super().__init__(sequences, node_feature_mappings, seed, node_idx_map, edge_idx_map, edge_types_inferred, no_external_constraints)

        self._sequence_info = [identify_sequence_constraints(self._filter_edge_ops(seq)) for seq in sequences]
        lengths = [info.num_stop_targets + info.num_constraint_targets for info in self._sequence_info]
        self._offsets = np.zeros(len(sequences) + 1, dtype=np.int32)
        np.cumsum(lengths, out=self._offsets[1:])

    def __len__(self):
        return self._offsets[-1]

    def __getitem__(self, index) -> AutoconstraintFeatureAndLabels:
        sequence_index = np.searchsorted(self._offsets, index, side='right') - 1
        target_index = index - self._offsets[sequence_index]

        seq_info = self._sequence_info[sequence_index]
        if target_index < seq_info.num_stop_targets:
            feature_info = extract_stop_target_features(seq_info, target_index)
        else:
            feature_info = extract_constraint_target_features(seq_info, target_index - seq_info.num_stop_targets)

        return self._make_features(seq_info, feature_info)


__all__ = [
    'NODE_TYPES', 'EDGE_TYPES', 'EDGE_TYPES_PREDICTED', 'NODE_IDX_MAP', 'EDGE_IDX_MAP',
    'EntityFeatureMapping', 'EdgeFeatureMapping', 'collate', 'AutoconstraintDataset'
]
