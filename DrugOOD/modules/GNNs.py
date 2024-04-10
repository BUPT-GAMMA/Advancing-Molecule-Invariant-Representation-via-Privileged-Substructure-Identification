import torch.nn as nn
import torch
import dgl.function as fn
import torch.nn.functional as F
from dgl.nn.pytorch import SumPooling, AvgPooling, MaxPooling, Set2Set, GlobalAttentionPooling
from drugood.models import BACKBONES

__all__ = ['GIN']



class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super(MLP, self).__init__()
        self.linear_or_not = True 
        self.num_layers = num_layers
        self.output_dim = output_dim

        if num_layers < 1:
            raise ValueError("number of layers should be positive!")
        elif num_layers == 1:
            self.linear = nn.Linear(input_dim, output_dim)
        else:
            self.linear_or_not = False
            self.linears = nn.ModuleList()
            self.bns = nn.ModuleList()

            self.linears.append(nn.Linear(input_dim, hidden_dim))
            for _ in range(num_layers - 2):
                self.linears.append(nn.Linear(hidden_dim, hidden_dim))
            self.linears.append(nn.Linear(hidden_dim, output_dim))

            for _ in range(num_layers - 1):
                self.bns.append(nn.BatchNorm1d(hidden_dim))

    def forward(self, x):
        if self.linear_or_not:
            return self.linear(x)
        else:
            h = x
            for i in range(self.num_layers - 1):
                h = F.relu(self.bns[i](self.linears[i](h)))
            return self.linears[-1](h)


class GINConv(nn.Module):
    def __init__(self, num_edge_emb_list, emb_dim, batch_norm = True, activation = None):
        super(GINConv, self).__init__()

        self.mlp = nn.Sequential(
            nn.Linear(emb_dim, 2 * emb_dim),
            nn.ReLU(),
            nn.Linear(2 * emb_dim, emb_dim)
        )
        self.edge_embs = nn.ModuleList()
        for num_emb in num_edge_emb_list:
            emb_module = MLP(input_dim = num_emb, hidden_dim = emb_dim,
                            output_dim = emb_dim, num_layers = 1)
            self.edge_embs.append(emb_module)                

        if batch_norm:
            self.bn = nn.BatchNorm1d(emb_dim)
        else:
            self.bn = None
        self.activation = activation
        self.reset_parameters()

    def reset_parameters(self):
        for layer in self.mlp:
            if isinstance(layer, nn.Linear):
                layer.reset_parameters()

        if self.bn is not None:
            self.bn.reset_parameters()

    def forward(self, g, node_feats, categorical_edge_feats):
        edge_embs = []
        for i, feats in enumerate(categorical_edge_feats):
            edge_embs.append(self.edge_embs[i](feats))
        edge_embs = torch.stack(edge_embs, dim = 0).sum(0)
        g = g.local_var()
        g.ndata['feat'] = node_feats
        g.edata['feat'] = edge_embs
        g.update_all(fn.u_add_e('feat', 'feat', 'm'), fn.sum('m', 'feat'))

        node_feats = self.mlp(g.ndata.pop('feat'))
        if self.bn is not None:
            node_feats = self.bn(node_feats)
        if self.activation is not None:
            node_feats = self.activation(node_feats)

        return node_feats


@BACKBONES.register_module()
class GIN(nn.Module):
    def __init__(self, num_node_emb_list, num_edge_emb_list,
                 num_layers=5, emb_dim=300, JK='last', dropout=0.5, readout="mean"):
        super(GIN, self).__init__()

        self.num_layers = num_layers
        self.JK = JK
        self.dropout = nn.Dropout(dropout)

        if num_layers < 2:
            raise ValueError('Number of GNN layers must be greater than 1, got {:d}'.format(num_layers))

        self.node_embs = nn.ModuleList()
        for num_emb in num_node_emb_list:
            emb_module = MLP(input_dim=num_emb, hidden_dim=emb_dim,
                             output_dim=emb_dim, num_layers=2)
            self.node_embs.append(emb_module)

        self.gnn_layers = nn.ModuleList()
        for layer in range(num_layers):
            if layer == num_layers - 1:
                self.gnn_layers.append(GINConv(num_edge_emb_list, emb_dim))
            else:
                self.gnn_layers.append(GINConv(num_edge_emb_list, emb_dim, activation=F.relu))

        if readout == 'sum':
            self.readout = SumPooling()
        elif readout == 'mean':
            self.readout = AvgPooling()
        elif readout == 'max':
            self.readout = MaxPooling()
        elif readout == 'attention':
            if JK == 'concat':
                self.readout = GlobalAttentionPooling(gate_nn=nn.Linear((num_layers + 1) * emb_dim, 1))
            else:
                self.readout = GlobalAttentionPooling(gate_nn=nn.Linear(emb_dim, 1))
        elif readout == 'set2set':
            self.readout = Set2Set()
        else:
            raise ValueError("Expect readout to be 'sum', 'mean', 'max', 'attention' or 'set2set', got {}".format(readout))

        self.reset_parameters()

    def reset_parameters(self):
        for layer in self.gnn_layers:
            layer.reset_parameters()

    def forward(self, input):
        categorical_node_feats = [input.ndata['x']]
        categorical_edge_feats = [input.edata['x']]

        node_embs = []
        for i, feats in enumerate(categorical_node_feats):
            node_embs.append(self.node_embs[i](feats))
        node_embs = torch.stack(node_embs, dim=0).sum(0)

        all_layer_node_feats = [node_embs]
        for layer in range(self.num_layers):
            node_feats = self.gnn_layers[layer](input, all_layer_node_feats[layer], categorical_edge_feats)
            node_feats = self.dropout(node_feats)
            all_layer_node_feats.append(node_feats)

        if self.JK == 'concat':
            final_node_feats = torch.cat(all_layer_node_feats, dim=1)
        elif self.JK == 'last':
            final_node_feats = all_layer_node_feats[-1]
        elif self.JK == 'max':
            all_layer_node_feats = [h.unsqueeze(0) for h in all_layer_node_feats]
            final_node_feats = torch.max(torch.cat(all_layer_node_feats, dim=0), dim=0)[0]
        elif self.JK == 'sum':
            all_layer_node_feats = [h.unsqueeze(0) for h in all_layer_node_feats]
            final_node_feats = torch.sum(torch.cat(all_layer_node_feats, dim=0), dim=0)
        else:
            return ValueError("Expect self.JK to be 'concat', 'last', 'max' or 'sum', got {}".format(self.JK))

        graph_feats = self.readout(input, final_node_feats)

        return graph_feats