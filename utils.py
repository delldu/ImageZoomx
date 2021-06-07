import os
import time
import shutil
import math

import torch
import numpy as np
from torch.optim import SGD, Adam
from tensorboardX import SummaryWriter
import pdb

class Averager():

    def __init__(self):
        self.n = 0.0
        self.v = 0.0

    def add(self, v, n=1.0):
        self.v = (self.v * self.n + v * n) / (self.n + n)
        self.n += n

    def item(self):
        return self.v


class Timer():

    def __init__(self):
        self.v = time.time()

    def s(self):
        self.v = time.time()

    def t(self):
        return time.time() - self.v


def time_text(t):
    if t >= 3600:
        return '{:.1f}h'.format(t / 3600)
    elif t >= 60:
        return '{:.1f}m'.format(t / 60)
    else:
        return '{:.1f}s'.format(t)


_log_path = None


def set_log_path(path):
    global _log_path
    _log_path = path


def log(obj, filename='log.txt'):
    print(obj)
    if _log_path is not None:
        with open(os.path.join(_log_path, filename), 'a') as f:
            print(obj, file=f)


def ensure_path(path, remove=True):
    basename = os.path.basename(path.rstrip('/'))
    if os.path.exists(path):
        if remove and (basename.startswith('_')
                or input('{} exists, remove? (y/[n]): '.format(path)) == 'y'):
            shutil.rmtree(path)
            os.makedirs(path)
    else:
        os.makedirs(path)


def set_save_path(save_path, remove=True):
    ensure_path(save_path, remove=remove)
    set_log_path(save_path)
    writer = SummaryWriter(os.path.join(save_path, 'tensorboard'))
    return log, writer


def compute_num_params(model, text=False):
    tot = int(sum([np.prod(p.shape) for p in model.parameters()]))
    if text:
        if tot >= 1e6:
            return '{:.1f}M'.format(tot / 1e6)
        else:
            return '{:.1f}K'.format(tot / 1e3)
    else:
        return tot


def make_optimizer(param_list, optimizer_spec, load_sd=False):
    Optimizer = {
        'sgd': SGD,
        'adam': Adam
    }[optimizer_spec['name']]
    optimizer = Optimizer(param_list, **optimizer_spec['args'])
    if load_sd:
        optimizer.load_state_dict(optimizer_spec['sd'])
    return optimizer


def make_coord(shape, flatten=True):
    """ Make coordinates at grid centers.
    """
    # pdb.set_trace()
    # shape = (384, 512)
    # ranges = None
    # flatten = True

    coord_seqs = []
    # for i, n in enumerate(shape): print(i, n)
    # 0 384
    # 1 512
    for i, n in enumerate(shape):
        v0, v1 = -1, 1
        r = (v1 - v0) / (2 * n)
        seq = v0 + r + (2 * r) * torch.arange(n).float()
        coord_seqs.append(seq)

        # (Pdb) torch.arange(10) -- tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        # pdb.set_trace()
        # i, n -- (0, 384), r -- 0.0026041666666666665, pp seq.size() -- torch.Size([384])
        # pp i, n -- (1, 512), r -- ...

    # (Pdb) len(coord_seqs), coord_seqs[0].size(), coord_seqs[1].size()
    # (2, torch.Size([384]), torch.Size([512]))

    ret = torch.stack(torch.meshgrid(*coord_seqs), dim=-1)
    # (Pdb) torch.stack(torch.meshgrid(*coord_seqs), dim=-1).size()
    # torch.Size([512, 960, 2])
    
    if flatten:
        ret = ret.view(-1, ret.shape[-1])

    # pdb.set_trace()
    # shape = (512, 960)
    # ranges = None
    # flatten = True

    # (Pdb) pp len(coord_seqs), coord_seqs[0].size(), coord_seqs[1].size()
    # (2, torch.Size([512]), torch.Size([960]))

    # coord_seqs[0]
    # tensor([-0.9980, -0.9941, -0.9902, -0.9863, -0.9824, -0.9785, -0.9746, -0.9707,
    # ...
    #          0.9707,  0.9746,  0.9785,  0.9824,  0.9863,  0.9902,  0.9941,  0.9980])


    # (Pdb) ret.size()
    # torch.Size([491520, 2]) -- 512 *960

    return ret


def to_pixel_samples(img):
    """ Convert the image to coord-RGB pairs.
        img: Tensor, (3, H, W)
    """
    coord = make_coord(img.shape[-2:])
    rgb = img.view(3, -1).permute(1, 0)
    pdb.set_trace()
    
    return coord, rgb


def calc_psnr(sr, hr, dataset=None, scale=1, rgb_range=1):
    diff = (sr - hr) / rgb_range
    if dataset is not None:
        if dataset == 'benchmark':
            shave = scale
            if diff.size(1) > 1:
                gray_coeffs = [65.738, 129.057, 25.064]
                convert = diff.new_tensor(gray_coeffs).view(1, 3, 1, 1) / 256
                diff = diff.mul(convert).sum(dim=1)
        elif dataset == 'div2k':
            shave = scale + 6
        else:
            raise NotImplementedError
        valid = diff[..., shave:-shave, shave:-shave]
    else:
        valid = diff
    mse = valid.pow(2).mean()
    return -10 * torch.log10(mse)
