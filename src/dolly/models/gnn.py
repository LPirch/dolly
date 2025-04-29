import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import Sequential, GINConv, global_add_pool
from torch_geometric.data import Data, Batch
from transformers import PretrainedConfig, PreTrainedModel

from dolly.models.base import CloneDetector


def _conv_block(in_channels, out_channels):
    linear = nn.Linear(in_channels, out_channels)
    return Sequential("x, edge_index", [
        (GINConv(linear), "x, edge_index -> x"),
        nn.Tanh(),
    ])


def _edge_convs(in_channels, out_channels, num_layers):
    conv_layers = []
    for i in range(num_layers):
        conv_layers.append(_conv_block(in_channels if i == 0 else out_channels, out_channels))
    return nn.ModuleList(conv_layers)


def _channel_mixing(hidden_size, edge_keys):
    num_channels = len(edge_keys)
    return nn.Sequential(
        nn.Linear(hidden_size*num_channels, hidden_size),
        nn.Tanh(),
        nn.BatchNorm1d(hidden_size),
        nn.Linear(hidden_size, hidden_size),
        nn.Tanh(),
        nn.BatchNorm1d(hidden_size),
    )


class GNNConfig(PretrainedConfig):
    model_type = "gnn"

    def __init__(
        self,
        in_channels=22,  # number of node types from node_type_map.json
        num_layers=5,
        hidden_size=128,
        classifier_dropout=0.1,
        edge_keys=["ast", "cfg", "dfg"],
        **kwargs
    ):
        super().__init__(**kwargs)
        self.in_channels = in_channels
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.classifier_dropout = classifier_dropout
        self.edge_keys = edge_keys


class GraphEncoder(PreTrainedModel):
    config_class = GNNConfig

    def __init__(self, config):
        super().__init__(config)
        self.config = config

        self.conv_layers = nn.ModuleDict()
        for key in config.edge_keys:
            self.conv_layers[key] = _edge_convs(
                config.in_channels,
                config.hidden_size,
                config.num_layers,
            )

        self.mixing_layers = nn.ModuleList([
            _channel_mixing(config.hidden_size, config.edge_keys)
            for _ in range(config.num_layers)
        ])
        self.pooling = global_add_pool

    def forward(self, x_types, batch, **edge_indices):
        x = F.one_hot(x_types, num_classes=self.config.in_channels)
        for layer in range(self.config.num_layers):
            _x = []
            for key in self.config.edge_keys:
                _x.append(self.conv_layers[key][layer](x, edge_indices[f"{key}_edge_index"]))
            x = self.mixing_layers[layer](torch.cat(_x, dim=-1))
        graph_embedding = self.pooling(x, batch, size=batch.max().item() + 1)
        return graph_embedding


class CloneGNN(CloneDetector):
    name = "gnn"
    signature_columns = ["a", "b", "label"]
    encoder_columns = ["x_types", "ast_edge_index", "cfg_edge_index", "dfg_edge_index"]
    base_encoder = "microsoft/unixcoder-base-nine"
    config_class = GNNConfig

    def __init__(self, config=None):
        if config is None:
            config = GNNConfig()
        encoder = GraphEncoder(config)
        super().__init__(encoder, config.pos_weight)

    def embed(self, sample):
        return self.encoder(**sample)

    @staticmethod
    def collate_fn(batch, tokenizer):
        batch = [_filter_columns(sample, CloneGNN.encoder_columns) for sample in batch]
        return Batch.from_data_list([Data(num_nodes=len(sample["x_types"]), **sample) for sample in batch])


def _filter_columns(sample, columns):
    return {k: v for k, v in sample.items() if k in columns}
