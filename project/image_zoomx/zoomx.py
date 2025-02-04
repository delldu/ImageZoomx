"""Create model."""  # coding=utf-8
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


def make_grid(H: int, W: int):
    """Make standard grid for H, W."""
    H = float(H)
    W = float(W)
    grid_h = torch.arange(-1.0, 1.0, 2.0 / H) + 1.0 / H
    grid_w = torch.arange(-1.0, 1.0, 2.0 / W) + 1.0 / W
    gh, gw = torch.meshgrid([grid_h, grid_w])
    # gh, gw size() -- [H, W]
    grid = torch.stack((gh.unsqueeze(0), gw.unsqueeze(0)), dim=1)
    # grid.size() -- torch.Size([1, 2, H, W])

    return grid


class NormalReLU(nn.Module):
    def __init__(self):
        """Init model."""
        super(NormalReLU, self).__init__()

    def forward(self, x):
        return torch.relu(x)


class ImageZoomxModel(nn.Module):
    """ImageZoomx Model."""

    def __init__(self):
        """Init model."""
        super(ImageZoomxModel, self).__init__()
        self.encoder = EDSR()

        in_dim = self.encoder.out_dim
        # self.encoder.out_dim -- 64
        in_dim *= 9
        in_dim += 2  # attach grid
        in_dim += 2  # attach cell
        out_dim = 3
        self.imnet = MLP(in_dim, out_dim, [256, 256, 256, 256])

    def forward(self, x, output_size):
        output_h = int(output_size[0])
        output_w = int(output_size[1])

        # output_h = 1024
        # output_w = 1024

        grid = make_grid(output_h, output_w)
        grid = grid.to(x.device)
        # grid.size() -- [1, 2, 1024, 1024]

        cell = torch.ones_like(grid)
        cell = torch.stack(
            (cell[:, 0, :, :] * 2.0 / output_h, cell[:, 1, :, :] * 2.0 / output_w),
            dim=1,
        )

        n = output_h * output_w
        grid = grid.view(1, 2, n, 1)
        # grid format from [1, 2, h, w] ==> [1, 2, h * w, 1]
        cell = cell.view(1, 2, n, 1)
        # cell format from [1, 2, h, w] ==> [1, 2, h * w, 1]

        bs: int = 256 * 256
        feat = self.encoder(x)

        # H, W = int(feat.shape[2]), int(feat.shape[3])
        H = feat.size(2)
        W = feat.size(3)

        feat_grid = make_grid(H, W)
        feat_grid = feat_grid.to(feat.device)
        # (Pdb) feat_grid.size() -- [1, 2, 96, 128]

        preds = []
        start: int = 0
        while start < n:
            # stop = min(start + bs, n)
            stop: int = start + bs if start + bs < n else n

            s_grid = grid[:, :, start:stop, :]
            s_cell = cell[:, :, start:stop, :]
            #  feat.size() -- [1, 576, 96, 128],
            #  s_grid.size(), s_cell.size()---- [1, 2, 65536, 1], [1, 2, 65536, 1]

            pred = self.imnet(feat, feat_grid, s_grid, s_cell)
            # pred.size() -- torch.Size([1, 3, 65536, 1])

            preds += [pred]
            start = stop

        # (Pdb) len(preds), preds[0].size(), preds[103].size(), preds[104].size()
        y = torch.cat(preds, dim=2)
        y = y[0].view(1, 3, output_h, output_w)

        return y.clamp(0, 1.0)


class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_list):
        # super().__init__()
        super(MLP, self).__init__()
        layers = []
        lastv = in_dim
        for hidden in hidden_list:
            layers.append(nn.Linear(lastv, hidden))
            layers.append(NormalReLU())
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

    def mlp_simple_forward(self, x):
        # x.size() -- torch.Size([bs, 580])
        bs = x.size(0)
        x = self.layers(x)
        return x.view(bs, -1)

    def forward(self, feat, feat_grid, s_grid, s_cell):
        # torch.Size([1, 576, 512, 512],
        # torch.Size([1, 2, 512, 512]),
        # torch.Size([1, 2, 65536, 1]),
        # torch.Size([1, 2, 65536, 1]))

        # s_grid, s_cell, format from [1, 2, 1, bs] to [1, bs, 2]
        s_grid = s_grid.squeeze(3).permute(0, 2, 1)
        s_cell = s_cell.squeeze(3).permute(0, 2, 1)
        # pp s_grid.size(), s_cell.size()
        # torch.Size([1, bs, 2]), torch.Size([1, bs, 2])

        batch = s_grid.size(0)
        chan = s_grid.size(1)

        B = feat.size(0)
        C = feat.size(1)
        H = feat.size(2)
        W = feat.size(3)
        # (Pdb) feat.size() --  [1, 576, 96, 128]

        eps_shift = 1e-6
        delta_h = 1.0 / H
        delta_w = 1.0 / W
        q_cell = s_cell.clone()
        q_cell = torch.stack((q_cell[:, :, 0] * H, q_cell[:, :, 1] * W), dim=2)

        # (Pdb) q_cell.size() -- torch.Size([1, bs, 2])
        # (Pdb) q_cell.min(), q_cell.max() --
        # (tensor(0.1250, device='cuda:0'), tensor(0.1250, device='cuda:0'))

        preds = []
        areas = []
        for r in range(2):
            for c in range(2):
                fine_grid = s_grid.clone()
                fine_grid = torch.stack(
                    (
                        fine_grid[:, :, 0] + (2.0 * r - 1.0) * delta_h,  # [-1.0, 1.0]
                        fine_grid[:, :, 1] + (2.0 * c - 1.0) * delta_w,  # [-1.0, 1.0]
                    ),
                    dim=2,
                )

                fine_grid = fine_grid.clamp(-1 + eps_shift, 1 - eps_shift)
                fine_grid = fine_grid.flip(-1).unsqueeze(1)

                # (Pdb) fine_grid.size()
                # torch.Size([1, 1, bs, 2])

                # q_feat = F.grid_sample(feat, fine_grid, mode='nearest',
                #     align_corners=False)[:, :, 0, :].permute(0, 2, 1)
                q_feat = torch.grid_sampler(feat, fine_grid, 1, 0, False)[:, :, 0, :].permute(0, 2, 1)

                # (Pdb) q_feat.size() -- torch.Size([1, bs, 576])

                # q_grid = F.grid_sample(feat_grid, fine_grid, mode='nearest',
                #     align_corners=False)[:, :, 0, :].permute(0, 2, 1)
                q_grid = torch.grid_sampler(feat_grid, fine_grid, 1, 0, False)[:, :, 0, :].permute(0, 2, 1)
                # (Pdb) q_grid.size() -- torch.Size([1, bs, 2])

                q_grid = s_grid - q_grid
                q_grid = torch.stack((q_grid[:, :, 0] * H, q_grid[:, :, 1] * W), dim=2)

                # (Pdb) q_grid.size() -- torch.Size([1, bs, 2])
                # (Pdb) q_grid.min(), q_grid.max()
                # -- (tensor(-0.9375, device='cuda:0'), tensor(1.9375, device='cuda:0'))

                # MLP Forward ...
                input = torch.cat([q_feat, q_grid, q_cell], dim=-1).view(batch * chan, -1)
                # input.size() -- torch.Size([bs, 580])

                pred = self.mlp_simple_forward(input).view(batch, chan, -1)

                # (Pdb) pred.size() -- torch.Size([1, bs, 3])
                preds += [pred]

                area = torch.abs(q_grid[:, :, 0] * q_grid[:, :, 1])
                areas += [area]

        total_area = torch.stack(areas).sum(dim=0) + 1e-9
        # t = areas[0], areas[0] = areas[3], areas[3] = t
        # t = areas[1], areas[1] = areas[2], areas[2] = t

        t_areas = []
        t_areas += [areas[3]]
        t_areas += [areas[2]]
        t_areas += [areas[1]]
        t_areas += [areas[0]]

        ret = torch.zeros_like(preds[0])
        for i in range(4):
            ret = ret + preds[i] * (t_areas[i] / total_area).unsqueeze(-1)

        # ret.size() -- torch.Size([1, bs, 3])  --> [1, 3, bs, 1]
        return ret.permute(0, 2, 1).unsqueeze(3)


def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(in_channels, out_channels, kernel_size, padding=(kernel_size // 2), bias=bias)


class MeanShift(nn.Conv2d):
    def __init__(
        self,
        rgb_range,
        rgb_mean=(0.4488, 0.4371, 0.4040),
        rgb_std=(1.0, 1.0, 1.0),
        sign=-1,
    ):
        super(MeanShift, self).__init__(3, 3, kernel_size=1)
        std = torch.Tensor(rgb_std)
        self.weight.data = torch.eye(3).view(3, 3, 1, 1) / std.view(3, 1, 1, 1)
        self.bias.data = sign * rgb_range * torch.Tensor(rgb_mean) / std
        for p in self.parameters():
            p.requires_grad = False


class ResBlock(nn.Module):
    def __init__(
        self,
        conv,
        n_feats,
        kernel_size,
        bias=True,
        bn=False,
        act=NormalReLU(),
        res_scale=1,
    ):
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
        # torch script error for self.res_scale is not Tensor !
        # res = self.body(x).mul(self.res_scale)
        res = self.body(x)
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
        self.sub_mean = MeanShift(rgb_range=1)
        self.add_mean = MeanShift(rgb_range=1, sign=1)

        # define head module
        m_head = [conv(n_colors, n_feats, kernel_size)]

        # define body module
        m_body = [ResBlock(conv, n_feats, kernel_size, act=NormalReLU(), res_scale=1) for i in range(n_resblocks)]
        m_body.append(conv(n_feats, n_feats, kernel_size))

        self.head = nn.Sequential(*m_head)
        self.body = nn.Sequential(*m_body)

        self.out_dim = n_feats

    def simple_forward(self, x):
        # x = self.sub_mean(x)
        x = self.head(x)
        res = self.body(x)
        return x + res

    def forward(self, x):
        x = self.simple_forward(x)

        # B, C, H, W = x.shape[0], x.shape[1], x.shape[2], x.shape[3]
        B = x.size(0)
        C = x.size(1)
        H = x.size(2)
        W = x.size(3)

        x = F.unfold(x, 3, dilation=1, padding=1, stride=1)
        # x = torch._C._nn.im2col(x, (3, 3), (1, 1), (1, 1), (1, 1))
        # x.view(B, C * 9, H, W) -- torch.Size([1, 576, 512, 512])

        return x.view(B, C * 9, H, W)
