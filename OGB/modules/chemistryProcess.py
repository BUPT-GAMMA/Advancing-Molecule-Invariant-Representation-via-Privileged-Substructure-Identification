from rdkit import Chem
from rdkit.Chem import BRICS, Recap
import numpy as np
from ogb.utils import smiles2graph
from torch_geometric.data import Data
import torch
from functools import reduce



def get_substructure(mol = None, smile = None, decomp = 'brics'):
    assert mol is not None or smile is not None, \
        'need at least one info of mol'
    assert decomp in ['brics', 'recap'], 'invalid decomposition method'
    if mol is None:
        mol = Chem.MolFromSmiles(smile)

    if decomp == 'brics':
        substructures = BRICS.BRICSDecompose(mol)
    else:
        recap_tree = Recap.RecapDecompose(mol)
        leaves = recap_tree.GetLeaves()
        substructures = set(leaves.keys())
    return substructures


def graph_from_substructure(subs, return_mask=False, return_type='numpy'):
    sub_struct_list = list(reduce(lambda x, y: x.update(y) or x, subs, set()))
    sub_to_idx = {x: idx for idx, x in enumerate(sub_struct_list)}
    mask = np.zeros([len(subs), len(sub_struct_list)], dtype=bool)
    sub_graph = [smiles2graph(x) for x in sub_struct_list]
    for idx, sub in enumerate(subs):
        mask[idx][list(sub_to_idx[t] for t in sub)] = True

    edge_idxes, edge_feats, node_feats, lstnode, batch = [], [], [], 0, []
    for idx, graph in enumerate(sub_graph):
        edge_idxes.append(graph['edge_index'] + lstnode)
        edge_feats.append(graph['edge_feat'])
        node_feats.append(graph['node_feat'])
        lstnode += graph['num_nodes']
        batch.append(np.ones(graph['num_nodes'], dtype=np.int64) * idx)

    result = {
        'edge_index': np.concatenate(edge_idxes, axis=-1),
        'edge_attr': np.concatenate(edge_feats, axis=0),
        'batch': np.concatenate(batch, axis=0),
        'x': np.concatenate(node_feats, axis=0)
    }

    assert return_type in ['numpy', 'torch', 'pyg'], 'Invaild return type'
    if return_type in ['torch', 'pyg']:
        for k, v in result.items():
            result[k] = torch.from_numpy(v)

    result['num_nodes'] = lstnode

    if return_type == 'pyg':
        result = Data(**result)

    if return_mask:
        return result, mask
    else:
        return result