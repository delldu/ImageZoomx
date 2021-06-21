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
from typing import List

import pdb

from torch.onnx.symbolic_helper import parse_args
from torch.onnx.symbolic_registry import register_op

@parse_args('v', 'v', 'i', 'i', 'i')
def grid_sampler(g, input, grid, interpolation_mode, padding_mode, align_corners=False):
    '''
    torch.nn.functional.grid_sample(input, grid, mode='bilinear', padding_mode='zeros', align_corners=None)
    Need convert interpolation_mode, padding_mode ? NO for simpler at now !!!
    '''
    return g.op('onnxservice::grid_sampler', input, grid,
        interpolation_mode_i=interpolation_mode,
        padding_mode_i=padding_mode,
        align_corners_i=align_corners)

register_op('grid_sampler', grid_sampler, '', 11)


@torch.jit.script
def make_grid(H:int, W:int):
    """Make standard grid for H, W."""
    grid_h = torch.arange(-1.0, 1.0, 2.0/H) + 1.0/H
    grid_w = torch.arange(-1.0, 1.0, 2.0/W) + 1.0/W

    grid = torch.stack(torch.meshgrid(grid_h, grid_w), dim=-1).unsqueeze(0)
    # grid.size() -- torch.Size([1, H, W, 2])

    return grid


class ImageZoomxModel(nn.Module):
    """ImageZoomx Model."""

    def __init__(self):
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

    def forward(self, x, output_size):
        '''
            output_size = torch.IntTensor([1024, 2048])
        '''

        output_height = int(output_size[0])
        output_width = int(output_size[1])

        grid = make_grid(output_height, output_width)
        grid = grid.to(x.device)

        cell = torch.ones_like(grid)
        # xxxx8888
        cell[:, :, :, 0] *= 2.0/output_height
        cell[:, :, :, 1] *= 2.0/output_width

        n = int(output_height * output_width)

        grid = grid.view(1, n, 2)
        # grid format from [1, h, w, 2] ==> [1, h * w, 2]
        cell = cell.view(1, n, 2)
        # cell format from [1, h, w, 2] ==> [1, h * w, 2]

        bs = 256 * 256
        feat = self.encoder(x)

        preds: List[torch.Tensor] = []
        start = 0
        while start < n:
            # stop = min(start + bs, n)
            stop = start + bs
            if stop > n:
                stop = n

            s_grid = grid[:, start: stop, :].unsqueeze(0)
            s_cell = cell[:, start: stop, :].unsqueeze(0)

            pred = self.imnet(feat, s_grid, s_cell)
            preds.append(pred)
            start = stop

        # (Pdb) len(preds), preds[0].size(), preds[103].size(), preds[104].size()
        # (105, torch.Size([1, bs, 3]), torch.Size([1, bs, 3]), torch.Size([1, bs, 3]))
        y = torch.cat(preds, dim=1)
        # pp y.size() -- torch.Size([1, 1048576, 3])
        y = y[0].view(1, output_height, output_width, 3).permute(0, 3, 1, 2)
        return y.clamp(0, 1.0)


class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_list: List[int]):
        super().__init__()
        layers: List[nn.Module()] = []
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
        # x.size() -- torch.Size([bs, 580])
        bs = x.shape[0]
        x = self.layers(x)
        return x.view(bs, -1)

    def forward(self, feat, s_grid, s_cell):
        # (Pdb) pp feat.size() -- torch.Size([1, 576, 96, 128])
        # (Pdb) pp s_grid.size() -- torch.Size([1, 1, 65536, 2])
        # (Pdb) s_cell.size() -- torch.Size([1, 1, 65536, 2])

        # s_grid, s_cell, format from [1, 1, bs, 2] to [1, bs, 2]
        s_grid = s_grid.squeeze(0)
        s_cell = s_cell.squeeze(0)

        # (Pdb) pp s_grid.size(), s_cell.size()
        # (torch.Size([1, bs, 2]), torch.Size([1, bs, 2]))
        batch, chan = s_grid.shape[:2]

        B, C, H, W = feat.shape[0], feat.shape[1], feat.shape[2], feat.shape[3]
        # (Pdb) feat.size() --  torch.Size([1, 576, 96, 128]
        feat_grid = make_grid(H, W).permute(0, 3, 1, 2)
        feat_grid = feat_grid.to(feat.device)
        # (Pdb) feat_grid.size() -- torch.Size([1, 2, 96, 128])

        eps_shift = 1e-6
        delta_h = 1 / H
        delta_w = 1 / W
        q_cell = s_cell.clone()
        # xxxx8888
        q_cell[:, :, 0] *= H
        q_cell[:, :, 1] *= W

        # (Pdb) q_cell.size() -- torch.Size([1, bs, 2])
        # (Pdb) q_cell.min(), q_cell.max() -- (tensor(0.1250, device='cuda:0'), tensor(0.1250, device='cuda:0'))

        preds: List[torch.Tensor] = []
        areas: List[torch.Tensor] = []
        for r in [-1, 1]:
            for c in [-1, 1]:
                fine_grid = s_grid.clone()

                # xxxx8888
                fine_grid[:, :, 0] += r * delta_h
                fine_grid[:, :, 1] += c * delta_w

                fine_grid = fine_grid.clamp(-1 + eps_shift, 1 - eps_shift)
                fine_grid = fine_grid.flip(-1).unsqueeze(1)

                # (Pdb) fine_grid.size()
                # torch.Size([1, 1, bs, 2])

                q_feat = F.grid_sample(feat, fine_grid, mode='nearest', align_corners=False)[:, :, 0, :].permute(0, 2, 1)
                # (Pdb) q_feat.size() -- torch.Size([1, bs, 576])

                q_grid = F.grid_sample(feat_grid, fine_grid, mode='nearest', align_corners=False)[:, :, 0, :].permute(0, 2, 1)
                # (Pdb) q_grid.size() -- torch.Size([1, bs, 2])

                q_grid = s_grid - q_grid
                # xxxx8888
                q_grid[:, :, 0] *= H
                q_grid[:, :, 1] *= W

                # (Pdb) q_grid.size() -- torch.Size([1, bs, 2])
                # (Pdb) q_grid.min(), q_grid.max() -- (tensor(-0.9375, device='cuda:0'), tensor(1.9375, device='cuda:0'))

                # MLP Forward ...
                input = torch.cat([q_feat, q_grid, q_cell], dim=-1).view(batch * chan, -1)
                # input.size() -- torch.Size([bs, 580])

                # with torch.no_grad():
                #     pred = self.simple_forward(input).view(batch, chan, -1)

                pred = self.simple_forward(input).view(batch, chan, -1)

                # (Pdb) pred.size() -- torch.Size([1, bs, 3])
                preds.append(pred)

                area = torch.abs(q_grid[:, :, 0] * q_grid[:, :, 1]) + 1e-9
                areas.append(area)

        total_area = torch.stack(areas).sum(dim=0)

        t = areas[0]; areas[0] = areas[3]; areas[3] = t
        t = areas[1]; areas[1] = areas[2]; areas[2] = t

        ret = torch.zeros_like(preds[0])
        # for pred, area in zip(preds, areas):
        #     ret = ret + pred * (area / total_area).unsqueeze(-1)
        for i in range(4):
            ret = ret + preds[i] * (areas[i] / total_area).unsqueeze(-1)

        # ret.size() -- torch.Size([1, bs, 3])
        return ret

def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(in_channels, out_channels, kernel_size, padding=(kernel_size//2), bias=bias)

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
        # xxxx8888, script building error for self.res_scale is not Tensor !
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
        self.sub_mean = MeanShift(rgb_range = 1)
        self.add_mean = MeanShift(rgb_range = 1, sign=1)

        # define head module
        m_head = [conv(n_colors, n_feats, kernel_size)]

        # define body module
        m_body = [
            ResBlock(
                conv, n_feats, kernel_size, act=nn.ReLU(True), res_scale=1
            ) for i in range(n_resblocks)
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


if __name__ == '__main__':
    model = ImageZoomxModel()
    # model = model.encoder

    script_model = torch.jit.script(model)

    print(script_model.code)
    print("-----------------------------------------------------")
    print(script_model.graph)
    print("-----------------------------------------------------")
    script_model.save("output/image_zoomx.th")

    input_image = torch.randn(1, 3, 256, 256)
    output_size = torch.Tensor([1024, 1024])

    output_image = script_model(input_image, output_size)

    print("output_image.size: ", output_image.size())
    print("-----------------------------------------------------")

    torch.onnx.export(script_model, (input_image, output_size), 'output/image_zoomx.onnx',
        opset_version=11, example_outputs=output_image, verbose=True, enable_onnx_checker=True)
