from copy import deepcopy
import numpy as np
from tqdm import tqdm
from torch_geometric.seed import seed_everything
import json
import time
import os
from sklearn.metrics import roc_auc_score
from mmcv import Config
from drugood.datasets import build_dataset, build_dataloader
from drugood.models import build_backbone
from torch.optim import AdamW
from modules.model import Framework
from modules.model import DeviationLoss
import argparse
from modules.utils import get_device, PenaltyWeightScheduler
from torch.nn.functional import binary_cross_entropy_with_logits as BCEWithLogitsLoss
import torch



def get_file_name(args):
    file_name = [f'lr_{args.lr}']
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
    current_time = time.time()
    file_dir = [f'{current_time}']
    return os.path.join(args.dataset, '-'.join(file_dir))


def init_args():
    parser = argparse.ArgumentParser('Parser for Experiments on DrugOOD')
    parser.add_argument(
        '--seed', default = 2023, type = int,
        help = 'random seed'
    )
    parser.add_argument(
        '--data_config', type = str, required = True,
        help = 'the path of data config'
    )
    parser.add_argument(
        '--encoder_config', type = str, required = True,
        help = 'the path of GNN encoder config'
    )
    parser.add_argument(
        '--lr', default = 0.0005, type = float,
        help = 'the learning rate of training'
    )
    parser.add_argument(
        '--exp_name', default = '', type = str,
        help = 'the name of logging file'
    )
    parser.add_argument(
        '--epoch_backbone', default = 120, type = int,
        help = 'the number of training epoch for backbone model'
    )
    parser.add_argument(
        '--dataset', default = 'ic50-assay', type = str,
        help = 'the dataset to run experiments'
    )
    parser.add_argument(
        '--device', default = 0, type = int,
        help = 'the gpu id for training'
    )

    args = parser.parse_args()
    args.work_dir = get_work_dir(args)
    if args.exp_name == '':
        args.exp_name, args.time = get_file_name(args)
    return args


def eval_one_epoch(loader, model, device, verbose=False):
    model = model.eval()
    y_pred, y_gt = [], []
    iterx = tqdm(loader) if verbose else loader
    for data in iterx:
        subs = [eval(x) for x in data['subs']]
        graphs = data['input'].to(device)
        labels = data['gt_label'].to(device)
        with torch.no_grad():
            preds = model(subs, graphs).squeeze()
        y_pred.append(preds.detach().cpu())
        y_gt.append(labels.detach().cpu())
    y_pred = torch.cat(y_pred, dim = 0).numpy()
    y_gt = torch.cat(y_gt, dim = 0).numpy()
    return roc_auc_score(y_gt, y_pred)


if __name__ == '__main__':
    args = init_args()
    print(args)
    if not os.path.exists('log'):
        os.mkdir('log')
    if not os.path.exists(os.path.join('log', args.work_dir)):
        os.makedirs(os.path.join('log', args.work_dir))
    seed_everything(args.seed)

    data_config = Config.fromfile(args.data_config)
    encoder_config = Config.fromfile(args.encoder_config)
    print(data_config.pretty_text)
    print(encoder_config.pretty_text)
    device = get_device(args.device)

    train_set = build_dataset(data_config.data.train)
    valid_set = build_dataset(data_config.data.ood_val)
    test_set = build_dataset(data_config.data.ood_test)
    data_config.data.ood_val.test_mode = True
    data_config.data.ood_test.test_mode = True
    train_loader = build_dataloader(
        train_set, data_config.data.samples_per_gpu,
        data_config.data.workers_per_gpu, num_gpus = 1,
        dist = False, round_up = True, seed = args.seed, shuffle = True
    )
    valid_loader = build_dataloader(
        valid_set, data_config.data.samples_per_gpu,
        data_config.data.workers_per_gpu, num_gpus = 1,
        dist = False, round_up = True, seed = args.seed, shuffle = False
    )
    test_loader = build_dataloader(
        test_set, data_config.data.samples_per_gpu,
        data_config.data.workers_per_gpu, num_gpus = 1,
        dist = False, round_up = True, seed = args.seed, shuffle = False
    )

    encoder = build_backbone(encoder_config.model.encoder)
    subencoder = build_backbone(encoder_config.model.subencoder)
    wrjModel = Framework(
        encoder = encoder, subencoder = subencoder,
        base_dim = encoder_config.model.encoder.emb_dim,
        sub_dim = encoder_config.model.subencoder.emb_dim,
        num_class = data_config.data.num_class, drop_ratio = encoder_config.drop_ratio
    ).to(device)
    
    optimizer = AdamW(wrjModel.parameters(), lr = args.lr)
    train_curv, valid_curv, test_curv = [], [], []
    best_model, best_reference_model = None, None
    best_valid, best_reference, best_ep = 0, 0, 0

    for ep in range(args.epoch_backbone):
        print(f'[INFO] Start training reference model on {ep} epoch')
        wrjModel = wrjModel.train()
        for data in tqdm(train_loader):
            subs = [eval(x) for x in data['subs']]
            graphs = data['input'].to(device)
            labels = data['gt_label'].to(device)

            preds = wrjModel(subs, graphs).squeeze()
            loss = BCEWithLogitsLoss(preds, labels.float())

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        print('[INFO] Evaluating the models')
        train_perf = eval_one_epoch(train_loader, wrjModel, device, verbose = True)
        valid_perf = eval_one_epoch(valid_loader, wrjModel, device, verbose = True)
        test_perf = eval_one_epoch(test_loader, wrjModel, device, verbose = True)

        if valid_perf > best_valid:
            best_valid = valid_perf
            best_model = deepcopy(wrjModel.state_dict())

        if train_perf - valid_perf > best_reference:
            best_reference = train_perf - valid_perf
            best_ep = ep
            best_reference_model = deepcopy(wrjModel.state_dict())

        train_curv.append(train_perf)
        valid_curv.append(valid_perf)
        test_curv.append(test_perf)
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