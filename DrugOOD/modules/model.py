import torch
import math
from drugood.core import move_to_device
from .utils import collect_batch_substrure_graphs
from torch.nn.functional import binary_cross_entropy_with_logits as BCEWithLogitsLoss

__all__ = ['Framework']



class Attention(torch.nn.Module):
    def __init__(self, Qdim, Kdim, Mdim):
        super(Attention, self).__init__()
        self.model_dim = Mdim
        self.WQ = torch.nn.Linear(Qdim, Mdim)
        self.WK = torch.nn.Linear(Qdim, Mdim)

    def forward(self, Q, K):
        Q, K = self.WQ(Q), self.WK(K)
        att = torch.matmul(Q, K.transpose(0, 1)) / math.sqrt(self.model_dim)
        return att


class Framework(torch.nn.Module):
    def __init__(
        self, encoder, subencoder, num_class,
        base_dim, sub_dim, drop_ratio = 0.5
    ):
        super(Framework, self).__init__()
        self.encoder = encoder
        self.subencoder = subencoder
        self.attn = Attention(
            base_dim, sub_dim, max(base_dim, sub_dim)
        )

        predictor_layers = [torch.nn.Linear(sub_dim, 2 * sub_dim)]
        if drop_ratio < 1 and drop_ratio > 0:
            predictor_layers.append(torch.nn.Dropout(drop_ratio))
        predictor_layers.append(torch.nn.ReLU())
        predictor_layers.append(torch.nn.Linear(2 * sub_dim, num_class))
        self.predictor_head = torch.nn.Sequential(*predictor_layers)

        environment_layers = [torch.nn.Linear(sub_dim, 2 * sub_dim)]
        if drop_ratio < 1 and drop_ratio > 0:
            environment_layers.append(torch.nn.Dropout(drop_ratio))
        environment_layers.append(torch.nn.ReLU())
        environment_layers.append(torch.nn.Linear(2 * sub_dim, num_class))
        self.environment_head = torch.nn.Sequential(*environment_layers)     

    def sub_feature_from_graphs(self, subs, device, return_mask = False):
        substructure_graph, mask = collect_batch_substrure_graphs(subs, True)
        substructure_graph = move_to_device(substructure_graph, device)
        substructure_feat = self.subencoder(substructure_graph)
        return (substructure_feat, mask) if return_mask else substructure_feat

    def forward(self, substructures, graphs, head = None, reverse_att = False):
        graph_feat = self.encoder(graphs)
        substructure_feat, att_mask = self.sub_feature_from_graphs(
            subs = substructures, device = graphs.device,
            return_mask = True
            )
        att_mask = torch.from_numpy(att_mask).to(graphs.device)
        att_mask = torch.logical_not(att_mask)
        att = self.attn(Q = graph_feat, K = substructure_feat)
        
        if head == 'environment':
            if reverse_att:
                att = -att
            if att_mask is not None:
                att = torch.masked_fill(att, att_mask, -(1 << 32))
            activation = torch.softmax(att, dim = -1)
            molecule_feat = torch.matmul(activation, substructure_feat)
            result = self.environment_head(molecule_feat)
            return result

        if att_mask is not None:
            att = torch.masked_fill(att, att_mask, -(1 << 32))
        activation = torch.softmax(att, dim = -1)
        molecule_feat = torch.matmul(activation, substructure_feat)
        result = self.predictor_head(molecule_feat)
        return result


def DeviationLoss(preds, y, device):
    scale = torch.tensor(1.).to(device).requires_grad_()
    loss = BCEWithLogitsLoss(preds * scale, y)
    grad = torch.autograd.grad(loss, [scale], create_graph = True)[0]
    return torch.sum(grad**2)

