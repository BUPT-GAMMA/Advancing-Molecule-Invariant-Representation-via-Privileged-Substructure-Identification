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
    file_name.append(f'ep_e_{args.epoch_env}')
    file_name.append(f'penalty_weight_{args.penalty_weight}')
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
        '--drop_ratio', default = 0.1, type = float,
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
        '--epoch_backbone', default = 60, type = int,
        help = 'the number of training epoch for backbone model'
    )
    parser.add_argument(
        '--epoch_env', default = 30, type = int,
        help = 'the number of training epoch for environment head'
    )
    parser.add_argument(
        '--adversarial_iteration', default = 6, type = int,
        help = 'the number of iteration for adversarial training'
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
        '--l2_weight', default = 1, type = float,
        help = 'the weight of l2 term'
    )
    parser.add_argument(
        '--penalty_weight', default = [0, 1, 200, 250, 250, 250, 250], type = list,
        help = 'the weight of penalty term in loss'
    )
    parser.add_argument(
        '--penalty_weight_ascend', default = 15, type = int,
        help = 'the number of anneal epoch'
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
            preds, att = model(subs, graphs)
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
    print(dataset.num_tasks)
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
    
    # pretrained reference model
    if not os.path.exists(f'pretrain/{args.dataset}'):
        raise IOError('please run pretrain script for dataset')

    pretrain_model = torch.load(f'pretrain/{args.dataset}/pretrain_reference.pt', map_location = device)
    sd = {}
    # for k, v in pretrain_model.items():
    for k, v in pretrain_model['backbone'].items():
        if 'predictor' in k:
            # sd[k] = v
            # sd[k.replace('predictor', 'environment')] = v
            sd[k.replace('predictor', 'predictor_head')] = v
            sd[k.replace('predictor', 'environment_head')] = v
        elif 'environment' in k:
            continue
        else:
            sd[k] = v
    # sd = torch.load(f'pretrain/{args.dataset}/prediction_iteration4.pt', map_location = device)
    wrjModel.load_state_dict(sd)

    best_model = None
    best_sum = 0
    for iteration in range(args.adversarial_iteration):
        if best_model:
            wrjModel.load_state_dict(best_model)
        env_params = []
        pred_params = []
        for k, v in wrjModel.named_parameters():
            if 'environment' in k:
                env_params.append(v)
            else:
                pred_params.append(v)
        optimizer_env = Adam(env_params, lr = args.lr, weight_decay = args.l2_weight)
        optimizer_pred = Adam(pred_params, lr = args.lr)
        train_env_curv = []
        best_env_model = None
        best_env_loss, best_env_ep = 0, 0

        # learning soft environment partition
        for ep in range(args.epoch_env):
            print(f'[INFO] Start training environment model on {ep} epoch')
            wrjModel = wrjModel.train()
            env_loss_batch = []
            for subs, graphs in tqdm(train_loader):
                subs = [eval(x) for x in subs]
                graphs = graphs.to(device)

                scale = torch.tensor(1.).to(device).requires_grad_()
                preds, _ = wrjModel(subs, graphs)
                loss = BCEWithLogitsLoss((preds * scale), graphs.y.float(), reduction = 'none')

                if iteration == 0:
                    envs = wrjModel(subs, graphs, head = 'environment')
                else:
                    envs = wrjModel(subs, graphs, head = 'environment', reverse_att = True)
                envs = envs.sigmoid().squeeze()
                labels = graphs.y.squeeze()

                idx00 = (labels == 0) & (envs < .5)
                idx01 = (labels == 1) & (envs > .5)
                idx10 = (labels == 0) & (envs > .5)
                idx11 = (labels == 1) & (envs < .5)
                loss = loss.squeeze()
                loss00 = loss[idx00] * envs[idx00]
                loss01 = loss[idx01] * (1 - envs[idx01])
                loss10 = loss[idx10] * (1 - envs[idx10])
                loss11 = loss[idx11] * envs[idx11]

                # penalty for environment one
                lossa = torch.cat([loss00, loss01]).mean()
                ## lossa = (loss.squeeze() * envs.sigmoid()).mean()
                # print(lossa)
                grada = torch.autograd.grad(lossa, [scale], create_graph = True)[0]
                penaltya = torch.sum(grada**2)
                # penalty for environment two
                lossb = torch.cat([loss10, loss11]).mean()
                ## lossb = (loss.squeeze() * (1 - envs.sigmoid())).mean()
                gradb = torch.autograd.grad(lossb, [scale], create_graph = True)[0]
                penaltyb = torch.sum(gradb**2)
                penalty = -torch.stack([penaltya, penaltyb]).mean()
                env_loss_batch.append(penalty.detach().cpu().numpy())

                optimizer_env.zero_grad()
                penalty.backward(retain_graph = True)
                optimizer_env.step()

            env_loss = np.mean(env_loss_batch)
            train_env_curv.append(float(env_loss))
            if env_loss < best_env_loss:
                best_env_loss = env_loss
                best_env_ep = ep
                best_env_model = deepcopy(wrjModel.state_dict())

        save_path = os.path.join('log', args.work_dir)
        torch.save(best_env_model, f'{save_path}/environment_iteration{iteration}.pt')

        print('Finished environment head training!')
        print('Best epoch: {}'.format(best_env_ep))
        print('Training loss: {}'.format(best_env_loss))

        wrjModel.load_state_dict(best_env_model)
        envModel = deepcopy(wrjModel)
        if iteration + 1 < len(args.penalty_weight):
            penalty_weight_scheduler = PenaltyWeightScheduler(epoch_to_max = args.penalty_weight_ascend, 
                init_val = args.penalty_weight[iteration], max_val = args.penalty_weight[iteration + 1])
        else:
            penalty_weight_scheduler = PenaltyWeightScheduler(epoch_to_max = args.penalty_weight_ascend, 
                init_val = args.penalty_weight[-1], max_val = args.penalty_weight[-1])

        train_curv, valid_curv, test_curv = [], [], []
        y_e1_ratio, y_e2_ratio = [], []
        e1_curv, e2_curv = [], []
        best_ep = -1
        for ep in range(args.epoch_backbone):
            print(f'[INFO] Start training prediction model on {ep} epoch')
            wrjModel = wrjModel.train()
            envModel = envModel.eval()
            y_e1_ratio_batch, y_e2_ratio_batch = [], []
            e1_curv_batch, e2_curv_batch = [], []
            for subs, graphs in tqdm(train_loader):
                subs = [eval(x) for x in subs]
                graphs = graphs.to(device)

                if iteration == 0:
                    envs = envModel(subs, graphs, head = 'environment')
                else:
                    envs = envModel(subs, graphs, head = 'environment', reverse_att = True)
                envs = envs.sigmoid().squeeze()
                labels = graphs.y.squeeze()

                idx00 = (labels == 0) & (envs < .5)
                idx01 = (labels == 1) & (envs > .5)
                idx10 = (labels == 0) & (envs > .5)
                idx11 = (labels == 1) & (envs < .5)
                idxa = idx00 + idx01
                idxb = idx10 + idx11
                ## idxa = (envs > .5)
                ## idxb = (envs <= .5)

                weights = torch.ones(len(graphs)).to(device)
                idx0 = ((labels == 0) & idxa)
                idx1 = ((labels == 1) & idxa)
                p0 = idx0.sum().float() / idxa.sum()
                p1 = 1 - p0
                y_e1_ratio_batch.append((p0/p1).detach().cpu().numpy())
                weights[idx0] = 1. / p0
                weights[idx1] = 1. / p1
                idx0 = ((labels == 0) & idxb)
                idx1 = ((labels == 1) & idxb)
                p0 = idx0.sum().float() / idxb.sum()
                p1 = 1 - p0
                y_e2_ratio_batch.append((p0/p1).detach().cpu().numpy())
                weights[idx0] = 1. / p0
                weights[idx1] = 1. / p1

                preds, _ = wrjModel(subs, graphs)
                lossa = BCEWithLogitsLoss(preds[idxa], graphs.y.float()[idxa], reduction = 'none')
                penaltya = DeviationLoss(preds[idxa], graphs.y.float()[idxa], device)
                lossb = BCEWithLogitsLoss(preds[idxb], graphs.y.float()[idxb], reduction = 'none')
                penaltyb = DeviationLoss(preds[idxb], graphs.y.float()[idxb], device)
                lossa = (weights[idxa].unsqueeze(1) * lossa).mean()
                lossb = (weights[idxb].unsqueeze(1) * lossb).mean()
                e1_curv_batch.append(lossa.detach().cpu().numpy())
                e2_curv_batch.append(lossb.detach().cpu().numpy())
                # lossa = lossa.mean()
                # lossb = lossb.mean()

                loss = torch.stack([lossa, lossb]).mean()
                penalty = torch.stack([penaltya, penaltyb]).mean()
                penalty_weight = penalty_weight_scheduler.step(ep)
                print(loss)
                loss += penalty_weight * penalty
                print(penalty_weight * penalty)

                optimizer_pred.zero_grad()
                loss.backward()
                optimizer_pred.step()

            y_e1_ratio.append(float(np.mean(y_e1_ratio_batch)))
            y_e2_ratio.append(float(np.mean(y_e2_ratio_batch)))
            e1_curv.append(float(np.mean(e1_curv_batch)))
            e2_curv.append(float(np.mean(e2_curv_batch)))
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

            if train_perf[dataset.eval_metric] + valid_perf[dataset.eval_metric] > best_sum:
                best_sum = train_perf[dataset.eval_metric] + valid_perf[dataset.eval_metric]
                best_ep = ep
                best_model = deepcopy(wrjModel.state_dict())

            train_curv.append(train_perf[dataset.eval_metric])
            valid_curv.append(valid_perf[dataset.eval_metric])
            test_curv.append(test_perf[dataset.eval_metric])
            print({'Train': train_perf, 'Valid': valid_perf, 'Test': test_perf})

        save_path = os.path.join('log', args.work_dir)
        torch.save(best_model, f'{save_path}/prediction_iteration{iteration}.pt')

        if best_ep != -1:
            print('Train score: {}'.format(train_curv[best_ep]))
            print('Validation score: {}'.format(valid_curv[best_ep]))
            print('Test score: {}'.format(test_curv[best_ep]))
            with open(os.path.join(save_path, args.exp_name), 'w') as f:
                json.dump({
                    'config': args.__dict__,
                    'train': train_curv,
                    'valid': valid_curv,
                    'test': test_curv,
                    'best': [
                        int(best_ep),
                        train_curv[best_ep],
                        valid_curv[best_ep],
                        test_curv[best_ep]
                    ],
                    'train_env': train_env_curv,
                    'y_e1_ratio': y_e1_ratio,
                    'y_e2_ratio': y_e2_ratio,
                    'e1_curv': e1_curv,
                    'e2_curv': e2_curv
                }, f)
        else:
            best_sum = 0

    wrjModel.load_state_dict(best_model)
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
    print('Finished training!')   
    print('Train score: {}'.format(train_perf[dataset.eval_metric]))
    print('Validation score: {}'.format(valid_perf[dataset.eval_metric]))
    print('Test score: {}'.format(test_perf[dataset.eval_metric]))