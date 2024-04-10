import torch
import numpy as np
from drugood.utils import smile2graph
from functools import reduce
import dgl



def get_device(number):
    return torch.device(f'cuda:{number}')


def collect_batch_substrure_graphs(subs, return_mask=True):
    sub_struct_list = list(reduce(lambda x, y: x.update(y) or x, subs, set()))
    sub_to_idx = {x: idx for idx, x in enumerate(sub_struct_list)}
    mask = np.zeros([len(subs), len(sub_struct_list)], dtype = bool)
    for idx, sub in enumerate(subs):
        mask[idx][list(sub_to_idx[t] for t in sub)] = True

    graphs = [smile2graph(x) for x in sub_struct_list]
    batch_data = dgl.batch(graphs)
    return (batch_data, mask) if return_mask else batch_data


class PenaltyWeightScheduler:
    def __init__(self, epoch_to_max, init_val, max_val):
        assert epoch_to_max >= 0
        self.epoch_to_max = epoch_to_max
        self.init_val = init_val
        self.max_val = max_val
        self.step_val = (self.max_val - self.init_val) / self.epoch_to_max if self.epoch_to_max > 0 else 0

    def step(self, epoch):
        if epoch < 0: 
            return self.init_val
        elif epoch >= self.epoch_to_max:
            return self.max_val
        else:
            return self.init_val + self.step_val * epoch