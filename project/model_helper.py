"""Create model."""# coding=utf-8
#
# /************************************************************************************
# ***
# ***    Copyright Dell 2021, All Rights Reserved.
# ***
# ***    File Author: Dell, 2021年 05月 27日 星期四 20:27:47 CST
# ***
# ************************************************************************************/
#

import pdb  # For debug

import torch
import torch.nn as nn
import torch.nn.functional as F

import pdb

@torch.jit.script
def make_grid(H:int, W:int, flatten:bool):
    """Make standard grid for H, W."""
    grid_h = torch.arange(-1.0, 1.0, 2.0/W) + 1.0/W
    grid_h = grid_h.view(1, 1, W).expand(-1, H, -1)

    grid_v = torch.arange(-1.0, 1.0, 2.0/H) + 1.0/H
    grid_v = grid_v.view(1, H, 1).expand(-1, -1, W)

    grid = torch.cat([grid_h, grid_v], dim=0)
    grid = grid.permute(1, 2, 0)
    # grid.size() -- torch.Size([H, W, 2])
    if flatten:
        grid = grid.view(-1, 2)

    return grid

class ImageZoomxModel(nn.Module):
    """ImageZoomx Model."""

    def __init__(self, h = 1024, w = 1024):
        """Init model."""

        super(ImageZoomxModel, self).__init__()
        self.encoder = EDSR()

        in_dim = self.encoder.out_dim
        # self.encoder.out_dim -- 64
        in_dim *= 9
        in_dim += 2 # attach grid
        in_dim += 2 # attach cell

        out_dim = 3
        self.imnet = MLP(in_dim, out_dim, [256, 256, 256, 256])
        self.reset_output_size(h, w)

    def reset_output_size(self, h, w):
        self.h = h
        self.w = w

        self.grid = make_grid(h, w, True)
        # self.grid -- [H * W, 2]
        self.cell = torch.ones_like(self.grid)
        self.cell[:, 0] *= 2 / self.h
        self.cell[:, 1] *= 2 / self.w

    def forward(self, x):
        with torch.no_grad():
            feat = self.encoder(x)

        grid = self.grid.unsqueeze(0)
        grid = grid.to(x.device)
        cell = self.cell.unsqueeze(0)
        cell = cell.to(x.device)

        bs = 128 * 128
        n = grid.shape[1]
        # (Pdb) grid.shape -- torch.Size([1, 491520, 2])

        start = 0
        preds = []
        while start < n:
            stop = min(start + bs, n)
            pred = self.imnet(feat, grid[:, start: stop, :], cell[:, start: stop, :])
            preds.append(pred)
            start = stop
        # (Pdb) len(preds), preds[0].size(), preds[103].size(), preds[104].size()
        # (105, torch.Size([1, 65536, 3]), torch.Size([1, 65536, 3]), torch.Size([1, 25728, 3]))
        y = torch.cat(preds, dim=1)
        # pp y.size() -- torch.Size([1, 1048576, 3])

        y = y[0]
        y = y.clamp(0, 1).view(self.h, self.w, 3).permute(2, 0, 1).cpu()

        return y


class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_list):
        super().__init__()
        layers = []
        lastv = in_dim
        for hidden in hidden_list:
            layers.append(nn.Linear(lastv, hidden))
            layers.append(nn.ReLU())
            lastv = hidden
        layers.append(nn.Linear(lastv, out_dim))
        self.layers = nn.Sequential(*layers)
        # self = MLP(
        #   (layers): Sequential(
        #     (0): Linear(in_features=580, out_features=256, bias=True)
        #     (1): ReLU()
        #     (2): Linear(in_features=256, out_features=256, bias=True)
        #     (3): ReLU()
        #     (4): Linear(in_features=256, out_features=256, bias=True)
        #     (5): ReLU()
        #     (6): Linear(in_features=256, out_features=256, bias=True)
        #     (7): ReLU()
        #     (8): Linear(in_features=256, out_features=3, bias=True)
        #   )
        # )
        # in_dim = 580
        # out_dim = 3
        # hidden_list = [256, 256, 256, 256]

    def simple_forward(self, x):
        # x.size() -- torch.Size([65536, 3])
        shape = x.shape[:-1]
        x = self.layers(x.view(-1, x.shape[-1]))
        # (Pdb) pp shape -- torch.Size([65536])
        return x.view(shape[0], -1)

    def forward(self, feat, s_grid, s_cell):
        # (Pdb) pp s_grid.size(), s_cell.size()
        # (torch.Size([1, 65536, 2]), torch.Size([1, 65536, 2]))

        B, C, H, W = feat.shape[0], feat.shape[1], feat.shape[2], feat.shape[3]
        batch, chan = s_grid.shape[:2]
        # (Pdb) feat.size(), s_grid.size(), s_cell.size()
        # (torch.Size([1, 576, 96, 128]), torch.Size([1, 65536, 2]), torch.Size([1, 65536, 2]))

        feat_grid = make_grid(H, W, False).permute(2, 0, 1).unsqueeze(0).expand(B, 2, H, W)
        feat_grid = feat_grid.to(feat.device)
        # feat_grid.size() -- torch.Size([1, 2, 96, 128])

        eps_shift = 1e-6
        delta_h = 1 / H
        delta_w = 1 / W
        q_cell = s_cell.clone()
        q_cell[:, :, 0] *= H
        q_cell[:, :, 1] *= W
        # (Pdb) q_cell.size() -- torch.Size([1, 65536, 2])
        # (Pdb) q_cell.min(), q_cell.max() -- (tensor(0.1250, device='cuda:0'), tensor(0.1250, device='cuda:0'))

        preds = []
        areas = []
        for r in [-1, 1]:
            for c in [-1, 1]:
                fine_grid = s_grid.clone()
                # fine_grid[:, :, 0] += r * delta_h + eps_shift
                # fine_grid[:, :, 1] += c * delta_w + eps_shift
                # fine_grid.clamp_(-1 + 1e-6, 1 - 1e-6)

                fine_grid[:, :, 0] += r * delta_h
                fine_grid[:, :, 1] += c * delta_w
                fine_grid.clamp_(-1 + eps_shift, 1 - eps_shift)
                fine_grid = fine_grid.flip(-1).unsqueeze(1)

                # (Pdb) fine_grid.size()
                # torch.Size([1, 1, 65536, 2])

                q_feat = F.grid_sample(feat, fine_grid, mode='nearest', align_corners=False)[:, :, 0, :].permute(0, 2, 1)
                # (Pdb) q_feat.size() -- torch.Size([1, 65536, 576])

                q_grid = F.grid_sample(feat_grid, fine_grid, mode='nearest', align_corners=False)[:, :, 0, :].permute(0, 2, 1)
                # (Pdb) q_grid.size() -- torch.Size([1, 65536, 2])

                q_grid = s_grid - q_grid
                q_grid[:, :, 0] *= H
                q_grid[:, :, 1] *= W
                # (Pdb) q_grid.size() -- torch.Size([1, 65536, 2])
                # (Pdb) q_grid.min(), q_grid.max() -- (tensor(-0.9375, device='cuda:0'), tensor(1.9375, device='cuda:0'))

                # MLP Forward ...
                input = torch.cat([q_feat, q_grid, q_cell], dim=-1).view(batch * chan, -1)
                # input.size() -- torch.Size([65536, 580])

                with torch.no_grad():
                    pred = self.simple_forward(input).view(batch, chan, -1)
                # (Pdb) pred.size() -- torch.Size([1, 65536, 3])
                preds.append(pred)

                area = torch.abs(q_grid[:, :, 0] * q_grid[:, :, 1])
                areas.append(area + 1e-9)

        total_area = torch.stack(areas).sum(dim=0)

        t = areas[0]; areas[0] = areas[3]; areas[3] = t
        t = areas[1]; areas[1] = areas[2]; areas[2] = t

        ret = 0
        for pred, area in zip(preds, areas):
            ret = ret + pred * (area / total_area).unsqueeze(-1)

        # # delete loop for onnx
        # ret += preds[0] * (areas[0]/total_area).unsqueeze(-1)
        # ret += preds[1] * (areas[1]/total_area).unsqueeze(-1)
        # ret += preds[2] * (areas[2]/total_area).unsqueeze(-1)
        # ret += preds[3] * (areas[3]/total_area).unsqueeze(-1)
        # (Pdb) areas[0].size() -- torch.Size([1, 65536])
        # preds[0].size() -- torch.Size([1, 65536, 3])

        return ret


def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size//2), bias=bias)

class MeanShift(nn.Conv2d):
    def __init__(
        self, rgb_range,
        rgb_mean=(0.4488, 0.4371, 0.4040), rgb_std=(1.0, 1.0, 1.0), sign=-1):

        super(MeanShift, self).__init__(3, 3, kernel_size=1)
        std = torch.Tensor(rgb_std)
        self.weight.data = torch.eye(3).view(3, 3, 1, 1) / std.view(3, 1, 1, 1)
        self.bias.data = sign * rgb_range * torch.Tensor(rgb_mean) / std
        for p in self.parameters():
            p.requires_grad = False

class ResBlock(nn.Module):
    def __init__(
        self, conv, n_feats, kernel_size,
        bias=True, bn=False, act=nn.ReLU(True), res_scale=1):

        super(ResBlock, self).__init__()
        m = []
        for i in range(2):
            m.append(conv(n_feats, n_feats, kernel_size, bias=bias))
            if bn:
                m.append(nn.BatchNorm2d(n_feats))
            if i == 0:
                m.append(act)

        self.body = nn.Sequential(*m)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x).mul(self.res_scale)
        res += x

        return res

class EDSR(nn.Module):
    def __init__(self, conv=default_conv):
        super(EDSR, self).__init__()
        n_resblocks = 16
        n_feats = 64
        kernel_size = 3
        scale = 2
        n_colors = 3
        self.sub_mean = MeanShift(rgb_range = 1)
        self.add_mean = MeanShift(rgb_range = 1, sign=1)

        # define head module
        m_head = [conv(n_colors, n_feats, kernel_size)]

        # define body module
        m_body = [
            ResBlock(
                conv, n_feats, kernel_size, act=nn.ReLU(True), res_scale=1
            ) for _ in range(n_resblocks)
        ]
        m_body.append(conv(n_feats, n_feats, kernel_size))

        self.head = nn.Sequential(*m_head)
        self.body = nn.Sequential(*m_body)

        self.out_dim = n_feats

    def simple_forward(self, x):
        #x = self.sub_mean(x)
        x = self.head(x)
        res = self.body(x)
        x += res
        #x = self.add_mean(x)
        return x

    def forward(self, x):
        x = self.simple_forward(x)

        B, C, H, W = x.shape[0], x.shape[1], x.shape[2], x.shape[3]
        x = F.unfold(x, 3, padding=1).view(B, C * 9, H, W)
        # pdb.set_trace()

        return x
