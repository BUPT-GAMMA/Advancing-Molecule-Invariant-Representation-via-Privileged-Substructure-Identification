import torch
from .GNNConv import GNN_node
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool, GlobalAttention



class GNNGraph(torch.nn.Module):
    def __init__(
        self, num_layer = 5, emb_dim = 300,
        gnn_type = 'gin', residual = False,
        drop_ratio = 0.5, JK = 'last', graph_pooling = 'mean'
    ):
        super(GNNGraph, self).__init__()

        self.num_layer = num_layer
        self.drop_ratio = drop_ratio
        self.JK = JK
        self.emb_dim = emb_dim
        self.graph_pooling = graph_pooling

        if self.num_layer < 2:
            raise ValueError('Number of GNN layers must be greater than 1.')

        self.gnn_node = GNN_node(
            num_layer, emb_dim, JK = JK, drop_ratio = drop_ratio,
            residual = residual, gnn_type = gnn_type
        )

        if self.graph_pooling == 'sum':
            self.pool = global_add_pool
        elif self.graph_pooling == 'mean':
            self.pool = global_mean_pool
        elif self.graph_pooling == 'max':
            self.pool = global_max_pool
        elif self.graph_pooling == 'attention':
            self.pool = GlobalAttention(gate_nn = torch.nn.Sequential(
                torch.nn.Linear(emb_dim, 2 * emb_dim),
                torch.nn.BatchNorm1d(2 * emb_dim),
                torch.nn.ReLU(),
                torch.nn.Linear(2 * emb_dim, 1)
                )
            )
        else:
            raise ValueError('Invalid graph pooling type.')

    def forward(self, batch_data):
        h_node = self.gnn_node(batch_data)

        h_graph = self.pool(h_node, batch_data.batch)
        return h_graph