from copy import deepcopy
import numpy as np
from tqdm import tqdm
from torch_geometric.loader import DataLoader
from torch_geometric.seed import seed_everything
import json
import time
import os
from modules.GNNs import GNNGraph
from modules.DataLoad import pyg_molsubdataset
from torch.optim import Adam
from modules.model import Framework
from modules.model import DeviationLoss
import argparse
from modules.utils import get_device, PenaltyWeightScheduler
from ogb.graphproppred import Evaluator
from torch.nn.functional import binary_cross_entropy_with_logits as BCEWithLogitsLoss
import torch



def get_file_name(args):
    file_name = [f'bs_{args.batch_size}']
    file_name.append(f'lr_{args.lr}')
    file_name.append(f'dr_{args.drop_ratio}')
    file_name.append(f'ep_bb_{args.epoch_backbone}')
    file_name.append(f'dataset_{args.dataset}')
    current_time = time.time()
    file_name.append(f'{current_time}')
    return '-'.join(file_name) + '.json', current_time


def get_basename(file_name):
    ans = os.path.basename(file_name)
    if '.' in ans:
        ans = ans.split('.')
        ans = '.'.join(ans[:-1])
    return ans


def get_work_dir(args):
    file_dir = [f'{get_basename(args.encoder)}']
    file_dir.append(f'{get_basename(args.subencoder)}')
    current_time = time.time()
    file_dir.append(f'{current_time}')
    return os.path.join(args.dataset, '-'.join(file_dir))


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
        '--seed', default = 2023, type = int,
        help = 'random seed'
    )
    parser.add_argument(
        '--encoder', type = str, required = True,
        help = 'the path of GNN encoder config'
    )
    parser.add_argument(
        '--subencoder', type = str, required = True,
        help = 'the path of GNN subencoder config'
    )
    parser.add_argument(
        '--drop_ratio', default = 0.1 , type = float,
        help = 'the dropout ratio of backbone model'
    )
    parser.add_argument(
        '--batch_size', default = 512, type = int,
        help = 'the batch size of training'
    )
    parser.add_argument(
        '--lr', default = 0.001, type = float,
        help = 'the learning rate of training'
    )
    parser.add_argument(
        '--exp_name', default = '', type = str,
        help = 'the name of logging file'
    )
    parser.add_argument(
        '--epoch_backbone', default = 100, type = int,
        help = 'the number of training epoch for backbone model'
    )
    parser.add_argument(
        '--dataset', default = 'ogbg-molbace', type = str,
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
    args.work_dir = get_work_dir(args)
    if args.exp_name == '':
        args.exp_name, args.time = get_file_name(args)
    return args


def eval_one_epoch(loader, evaluator, model, device, verbose=False):
    model = model.eval()
    y_pred, y_gt = [], []
    iterx = tqdm(loader) if verbose else loader
    for subs, graphs in iterx:
        subs = [eval(x) for x in subs]
        graphs = graphs.to(device)
        with torch.no_grad():
            preds, _  = model(subs, graphs)
        y_pred.append(preds.detach().cpu())
        y_gt.append(graphs.y.reshape(preds.shape).detach().cpu())
    y_pred = torch.cat(y_pred, dim = 0).numpy()
    y_gt = torch.cat(y_gt, dim = 0).numpy()
    return evaluator.eval({'y_true': y_gt, 'y_pred': y_pred})


if __name__ == '__main__':
    args = init_args()
    print(args)
    if not os.path.exists('log'):
        os.mkdir('log')
    if not os.path.exists(os.path.join('log', args.work_dir)):
        os.makedirs(os.path.join('log', args.work_dir))
    seed_everything(args.seed)

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
        num_class = dataset.num_tasks, drop_ratio = args.drop_ratio
    ).to(device)

    evaluator = Evaluator(args.dataset)
    data_split_idx = dataset.get_idx_split()
    train_idx = data_split_idx['train']
    valid_idx = data_split_idx['valid']
    test_idx = data_split_idx['test']

    train_dataset = dataset[train_idx]
    valid_dataset = dataset[valid_idx]
    test_dataset = dataset[test_idx]
    train_subs = [str(total_subs[x.item()]) for x in train_idx]
    valid_subs = [str(total_subs[x.item()]) for x in valid_idx]
    test_subs = [str(total_subs[x.item()]) for x in test_idx]

    train_loader = DataLoader(
        list(zip(train_subs, train_dataset)),
        batch_size = args.batch_size, shuffle = True
    )
    valid_loader = DataLoader(
        list(zip(valid_subs, valid_dataset)),
        batch_size = args.batch_size, shuffle = False
    )
    test_loader = DataLoader(
        list(zip(test_subs, test_dataset)),
        batch_size = args.batch_size, shuffle = False
    )
    
    optimizer = Adam(wrjModel.parameters(), lr = args.lr)
    train_curv, valid_curv, test_curv = [], [], []
    best_model, best_reference_model = None, None
    best_valid, best_reference, best_ep = 0, 0, 0

    for ep in range(args.epoch_backbone):
        print(f'[INFO] Start training reference model on {ep} epoch')
        wrjModel = wrjModel.train()
        for subs, graphs in tqdm(train_loader):
            subs = [eval(x) for x in subs]
            graphs = graphs.to(device)

            preds, _ = wrjModel(subs, graphs)
            loss = BCEWithLogitsLoss(preds, graphs.y.float())

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        print('[INFO] Evaluating the models')
        train_perf = eval_one_epoch(
            train_loader, evaluator, wrjModel,
            device, verbose = True
        )
        valid_perf = eval_one_epoch(
            valid_loader, evaluator, wrjModel,
            device, verbose = True
        )
        test_perf = eval_one_epoch(
            test_loader, evaluator, wrjModel,
            device, verbose = True
        )

        if valid_perf[dataset.eval_metric] > best_valid:
            best_valid = valid_perf[dataset.eval_metric]
            best_model = deepcopy(wrjModel.state_dict())

        if train_perf[dataset.eval_metric] - valid_perf[dataset.eval_metric] > best_reference:
            best_reference = train_perf[dataset.eval_metric] - valid_perf[dataset.eval_metric]
            best_ep = ep
            best_reference_model = deepcopy(wrjModel.state_dict())

        train_curv.append(train_perf[dataset.eval_metric])
        valid_curv.append(valid_perf[dataset.eval_metric])
        test_curv.append(test_perf[dataset.eval_metric])
        print({'Train': train_perf, 'Valid': valid_perf, 'Test': test_perf})

    save_path = os.path.join('log', args.work_dir)
    torch.save(best_model, f'{save_path}/pretrain_pred.pt')
    torch.save(best_reference_model, f'{save_path}/pretrain_reference.pt')

    best_val_epoch = np.argmax(np.array(valid_curv))
    print('Finished training!')
    print('Best epoch: {}'.format(best_val_epoch))
    print('Train score: {}'.format(train_curv[best_val_epoch]))
    print('Validation score: {}'.format(valid_curv[best_val_epoch]))
    print('Test score: {}'.format(test_curv[best_val_epoch]))
    with open(os.path.join(save_path, args.exp_name), 'w') as f:
        json.dump({
            'config': args.__dict__,
            'train': train_curv,
            'valid': valid_curv,
            'test': test_curv,
            'best': [
                int(best_val_epoch),
                train_curv[best_val_epoch],
                valid_curv[best_val_epoch],
                test_curv[best_val_epoch]
            ],
        }, f)

    print('>> ******************************* <<')
    print('Best diff epoch: {}'.format(best_ep))
    print('Diff score: {}'.format(train_curv[best_ep] - valid_curv[best_ep]))
    print('Train score: {}'.format(train_curv[best_ep]))
    print('Validation score: {}'.format(valid_curv[best_ep]))
    print('Test score: {}'.format(test_curv[best_ep]))