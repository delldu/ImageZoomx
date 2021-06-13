import torch
import torch.nn as nn
import torch.nn.functional as F

import models
from models import register
from utils import make_coord
import pdb

@register('liif')
class LIIF(nn.Module):

    def __init__(self, encoder_spec, imnet_spec=None,
                 local_ensemble=True, feat_unfold=True, cell_decode=True):
        super().__init__()
        self.local_ensemble = local_ensemble
        self.feat_unfold = feat_unfold
        self.cell_decode = cell_decode

        # encoder_spec = {'name': 'edsr-baseline', 'args': {'no_upsampling': True}}
        self.encoder = models.make(encoder_spec)

        # pdb.set_trace()
        # imnet_spec
        # {'name': 'mlp', 'args': {'out_dim': 3, 'hidden_list': [256, 256, 256, 256]}}
        if imnet_spec is not None:
            imnet_in_dim = self.encoder.out_dim
            # self.encoder.out_dim -- 64

            # self.feat_unfold -- True
            if self.feat_unfold:
                imnet_in_dim *= 9
            imnet_in_dim += 2 # attach grid
            if self.cell_decode:
                imnet_in_dim += 2
            self.imnet = models.make(imnet_spec, args={'in_dim': imnet_in_dim})
        else:
            self.imnet = None
        # (Pdb) imnet_in_dim -- 580

        # imnet_spec = {'name': 'mlp', 'args': {'out_dim': 3, 'hidden_list': [256, 256, 256, 256]}}
        # local_ensemble = True
        # feat_unfold = True
        # cell_decode = True

        # (Pdb) self.imnet
        # MLP(
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


    def gen_feat(self, input):
        self.feat = self.encoder(input)
        # (Pdb) input.size()
        # torch.Size([1, 3, 384, 510])
        # images/0803.png: PNG image data, 510 x 384, 8-bit/color RGB, non-interlaced
        # (Pdb) self.feat.size() -- torch.Size([1, 576, 384, 510])
        return self.feat

    def query_rgb(self, s_grid, s_cell):
        # (Pdb) pp s_grid.size(), s_cell.size()
        # (torch.Size([1, 30000, 2]), torch.Size([1, 30000, 2]))

        feat = self.feat
        B, C, H, W = feat.shape[0], feat.shape[1], feat.shape[2], feat.shape[3]
        batch, chan = s_grid.shape[:2]
        # (Pdb) feat.size(), s_grid.size(), s_cell.size()
        # (torch.Size([1, 576, 96, 128]), torch.Size([1, 30000, 2]), torch.Size([1, 30000, 2]))

        feat_grid = make_coord((H, W), flatten=False).cuda() \
            .permute(2, 0, 1) \
            .unsqueeze(0).expand(B, 2, H, W)
        # feat_grid.size() -- torch.Size([1, 2, 96, 128])

        eps_shift = 1e-6
        delta_h = 1 / H
        delta_w = 1 / W
        q_cell = s_cell.clone()
        q_cell[:, :, 0] *= H
        q_cell[:, :, 1] *= W
        # (Pdb) q_cell.size() -- torch.Size([1, 30000, 2])
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
                # torch.Size([1, 30000, 2])

                # (Pdb) x = torch.randn(1, 3, 2)
                # (Pdb) x
                # tensor([[[-0.8006,  0.5135],
                #          [-0.1273, -0.0481],
                #          [-1.1217,  1.5947]]])
                # (Pdb) x.flip(-1)
                # tensor([[[ 0.5135, -0.8006],
                #          [-0.0481, -0.1273],
                #          [ 1.5947, -1.1217]]])

                # F.grid_sample(feat, fine_grid.flip(-1).unsqueeze(1),mode='nearest', align_corners=False).size() --
                # torch.Size([1, 576, 1, 65536])
                q_feat = F.grid_sample(feat, fine_grid, mode='nearest', align_corners=False)[:, :, 0, :].permute(0, 2, 1)

                # (Pdb) q_feat.size() -- torch.Size([1, 30000, 576])

                q_grid = F.grid_sample(feat_grid, fine_grid, mode='nearest', align_corners=False)[:, :, 0, :].permute(0, 2, 1)
                # (Pdb) q_grid.size() -- torch.Size([1, 30000, 2])

                q_grid = s_grid - q_grid
                q_grid[:, :, 0] *= H
                q_grid[:, :, 1] *= W
                # (Pdb) q_grid.size() -- torch.Size([1, 30000, 2])
                # (Pdb) q_grid.min(), q_grid.max() -- (tensor(-0.9375, device='cuda:0'), tensor(1.9375, device='cuda:0'))

                # MLP Forward ...
                input = torch.cat([q_feat, q_grid, q_cell], dim=-1).view(batch * chan, -1)
                # input.size() -- torch.Size([30000, 580])

                pred = self.imnet(input).view(batch, chan, -1)
                # (Pdb) pred.size() -- torch.Size([1, 30000, 3])
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

    def forward(self, input, grid, cell):
        pdb.set_trace()

        self.gen_feat(input)
        return self.query_rgb(grid, cell)
