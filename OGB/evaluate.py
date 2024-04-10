from tqdm import tqdm
from torch_geometric.loader import DataLoader
import json
import os
from modules.GNNs import GNNGraph
from modules.DataLoad import pyg_molsubdataset
from modules.model import Framework
import argparse
from modules.utils import get_device
from ogb.graphproppred import Evaluator
import torch



def build_model_from_config(config):
    model_type = config['type']
    if model_type == 'gin':
        model = GNNGraph(gnn_type = 'gin', **config['params'])
    else:
        raise ValueError(f'Invalid model type called {model_type}')
    return model


def init_args():
    parser = argparse.ArgumentParser('Parser for Experiments on OGB')
    parser.add_argument(
        '--encoder', type = str, required = True,
        help = 'the path of GNN encoder config'
    )
    parser.add_argument(
        '--subencoder', type = str, required = True,
        help = 'the path of GNN subencoder config'
    )
    parser.add_argument(
        '--batch_size', default = 1, type = int,
        help = 'the batch size of training'
    )
    parser.add_argument(
        '--exp_name', default = '', type = str,
        help = 'the name of logging file'
    )
    parser.add_argument(
        '--dataset', default = 'ogbg-molbbbp', type = str,
        help = 'the dataset to run experiments'
    )
    parser.add_argument(
        '--device', default = 0, type = int,
        help = 'the gpu id for training'
    )
    parser.add_argument(
        '--decomp', choices = ['brics', 'recap'], default = 'brics',
        help = 'the method to decompose the molecules into substructures'
    )

    args = parser.parse_args()
    return args


def eval_one_epoch(loader, evaluator, model, device, verbose=False):
    model = model.eval()
    y_pred, y_gt = [], []
    iterx = tqdm(loader) if verbose else loader
    idx = 1
    for smiles, subs, graphs in iterx:
        subs = [eval(x) for x in subs]
        graphs = graphs.to(device)
        with torch.no_grad():
            preds, att = model(subs, graphs)
        y_pred.append(preds.detach().cpu())
        y_gt.append(graphs.y.reshape(preds.shape).detach().cpu())
        if preds.sigmoid() > 0.999 and graphs.num_nodes > 24:
            print(graphs, smiles, subs, att)
            print(idx, preds.sigmoid(), graphs.y.reshape(preds.shape))
        idx += 1
    y_pred = torch.cat(y_pred, dim = 0).numpy()
    y_gt = torch.cat(y_gt, dim = 0).numpy()
    return evaluator.eval({'y_true': y_gt, 'y_pred': y_pred})


if __name__ == '__main__':
    args = init_args()
    print(args)

    with open(args.encoder) as f:
        encoder_config = json.load(f)
    with open(args.subencoder) as f:
        subencoder_config = json.load(f)

    total_smiles, total_subs, dataset = pyg_molsubdataset(
        args.dataset, args.decomp
    )
    device = get_device(args.device)

    encoder = build_model_from_config(encoder_config)
    subencoder = build_model_from_config(subencoder_config)
    wrjModel = Framework(
        encoder = encoder, subencoder = subencoder,
        base_dim = encoder_config['result_dim'],
        sub_dim = subencoder_config['result_dim'],
        num_class = dataset.num_tasks
    ).to(device)

    evaluator = Evaluator(args.dataset)
    data_split_idx = dataset.get_idx_split()
    # train_idx = data_split_idx['train']
    # valid_idx = data_split_idx['valid']
    test_idx = data_split_idx['test']

    # train_dataset = dataset[train_idx]
    # valid_dataset = dataset[valid_idx]
    test_dataset = dataset[test_idx]
    # train_subs = [str(total_subs[x.item()]) for x in train_idx]
    # valid_subs = [str(total_subs[x.item()]) for x in valid_idx]
    test_smiles = [str(total_smiles[x.item()]) for x in test_idx]
    test_subs = [str(total_subs[x.item()]) for x in test_idx]

    # train_loader = DataLoader(
    #     list(zip(train_subs, train_dataset)),
    #     batch_size = args.batch_size, shuffle = True
    # )
    # valid_loader = DataLoader(
    #     list(zip(valid_subs, valid_dataset)),
    #     batch_size = args.batch_size, shuffle = False
    # )
    test_loader = DataLoader(
        list(zip(test_smiles, test_subs, test_dataset)),
        batch_size = args.batch_size, shuffle = False
    )

    pretrain_model = torch.load(f'pretrain/{args.dataset}/prediction_iteration2.pt', map_location = device)
    wrjModel.load_state_dict(pretrain_model)

    print('[INFO] Evaluating the models')
    # train_perf = eval_one_epoch(
    #     train_loader, evaluator, wrjModel,
    #     device, verbose = True
    # )
    # valid_perf = eval_one_epoch(
    #     valid_loader, evaluator, wrjModel,
    #     device, verbose = True
    # )
    test_perf = eval_one_epoch(
        test_loader, evaluator, wrjModel,
        device, verbose = True
    )
 
    print('Finished training!')   
    # print('Train score: {}'.format(train_perf[dataset.eval_metric]))
    # print('Validation score: {}'.format(valid_perf[dataset.eval_metric]))
    print('Test score: {}'.format(test_perf[dataset.eval_metric]))