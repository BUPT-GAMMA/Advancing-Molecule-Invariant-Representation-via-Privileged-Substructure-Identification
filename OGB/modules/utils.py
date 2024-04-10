import torch



def get_device(number):
    return torch.device(f'cuda:{number}')


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