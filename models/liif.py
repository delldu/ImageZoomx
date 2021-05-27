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
            imnet_in_dim += 2 # attach coord
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


    def gen_feat(self, inp):
        self.feat = self.encoder(inp)
        # (Pdb) inp.size()
        # torch.Size([1, 3, 384, 510])
        # images/0803.png: PNG image data, 510 x 384, 8-bit/color RGB, non-interlaced
        # (Pdb) self.feat.size() -- torch.Size([1, 64, 384, 510])

        return self.feat

    def query_rgb(self, coord, cell=None):
        feat = self.feat

        # pdb.set_trace()
        # (Pdb) feat.size(), coord.size(), cell.size()
        # (torch.Size([1, 64, 96, 128]), torch.Size([1, 30000, 2]), torch.Size([1, 30000, 2]))

        if self.imnet is None:
            ret = F.grid_sample(feat, coord.flip(-1).unsqueeze(1),
                mode='nearest', align_corners=False)[:, :, 0, :] \
                .permute(0, 2, 1)
            return ret

        # self.feat_unfold -- True
        if self.feat_unfold:
            # feat.shape:  torch.Size([1, 64, 96, 128])
            feat = F.unfold(feat, 3, padding=1).view(
                feat.shape[0], feat.shape[1] * 9, feat.shape[2], feat.shape[3])

        # self.local_ensemble -- True
        if self.local_ensemble:
            vx_lst = [-1, 1]
            vy_lst = [-1, 1]
            eps_shift = 1e-6
        else:
            vx_lst, vy_lst, eps_shift = [0], [0], 0

        # field radius (global: [-1, 1])
        rx = 2 / feat.shape[-2] / 2
        ry = 2 / feat.shape[-1] / 2
        # (Pdb) pp rx, ry -- (0.010416666666666666, 0.0078125)

        # (Pdb) feat.shape -- torch.Size([1, 576, 96, 128])
        # (Pdb) feat.shape[-2:] -- torch.Size([96, 128])
        feat_coord = make_coord(feat.shape[-2:], flatten=False).cuda() \
            .permute(2, 0, 1) \
            .unsqueeze(0).expand(feat.shape[0], 2, *feat.shape[-2:])
        # pdb.set_trace()
        # (Pdb) pp coord.size(), cell.size()
        # (torch.Size([1, 30000, 2]), torch.Size([1, 30000, 2]))

        preds = []
        areas = []
        for vx in vx_lst:
            for vy in vy_lst:
                coord_ = coord.clone()
                coord_[:, :, 0] += vx * rx + eps_shift
                coord_[:, :, 1] += vy * ry + eps_shift
                coord_.clamp_(-1 + 1e-6, 1 - 1e-6)
                q_feat = F.grid_sample(
                    feat, coord_.flip(-1).unsqueeze(1),
                    mode='nearest', align_corners=False)[:, :, 0, :] \
                    .permute(0, 2, 1)
                q_coord = F.grid_sample(
                    feat_coord, coord_.flip(-1).unsqueeze(1),
                    mode='nearest', align_corners=False)[:, :, 0, :] \
                    .permute(0, 2, 1)
                rel_coord = coord - q_coord
                rel_coord[:, :, 0] *= feat.shape[-2]
                rel_coord[:, :, 1] *= feat.shape[-1]
                inp = torch.cat([q_feat, rel_coord], dim=-1)
                # self.cell_decode -- True
                if self.cell_decode:
                    rel_cell = cell.clone()
                    rel_cell[:, :, 0] *= feat.shape[-2]
                    rel_cell[:, :, 1] *= feat.shape[-1]
                    inp = torch.cat([inp, rel_cell], dim=-1)

                bs, q = coord.shape[:2]
                # MLP Forward ...
                # print("--- inp:", inp.size(), "bs:", bs, "q:", q)
                # inp: torch.Size([1, 30000, 580]) bs: 1 q: 30000

                pred = self.imnet(inp.view(bs * q, -1)).view(bs, q, -1)
                preds.append(pred)

                area = torch.abs(rel_coord[:, :, 0] * rel_coord[:, :, 1])
                areas.append(area + 1e-9)

        tot_area = torch.stack(areas).sum(dim=0)

        # self.local_ensemble -- True
        if self.local_ensemble:
            t = areas[0]; areas[0] = areas[3]; areas[3] = t
            t = areas[1]; areas[1] = areas[2]; areas[2] = t
        ret = 0
        for pred, area in zip(preds, areas):
            ret = ret + pred * (area / tot_area).unsqueeze(-1)
        return ret

    def forward(self, inp, coord, cell):
        pdb.set_trace()

        self.gen_feat(inp)
        return self.query_rgb(coord, cell)
