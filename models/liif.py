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

        # feat.shape:  torch.Size([1, 64, 96, 128])
        feat = F.unfold(feat, 3, padding=1).view(feat.shape[0], feat.shape[1] * 9, feat.shape[2], feat.shape[3])

        eps_shift = 1e-6

        # field radius (global: [-1, 1])
        delta_h = 2 / feat.shape[-2] / 2
        delta_w = 2 / feat.shape[-1] / 2
        # (Pdb) pp delta_h, delta_w -- (0.010416666666666666, 0.0078125)

        # (Pdb) feat.shape -- torch.Size([1, 576, 96, 128])
        # (Pdb) feat.shape[-2:] -- torch.Size([96, 128])
        feat_coord = make_coord(feat.shape[-2:], flatten=False).cuda() \
            .permute(2, 0, 1) \
            .unsqueeze(0).expand(feat.shape[0], 2, *feat.shape[-2:])
        # feat_coord.size() -- torch.Size([1, 2, 96, 128])


        # (Pdb) pp coord.size(), cell.size()
        # (torch.Size([1, 30000, 2]), torch.Size([1, 30000, 2]))


        # for r in [-1, 1]: print(r)
        # -1, 1

        preds = []
        areas = []
        for r in [-1, 1]:
            for c in [-1, 1]:
                coord_ = coord.clone()
                # coord_[:, :, 0] += r * delta_h + eps_shift
                # coord_[:, :, 1] += c * delta_w + eps_shift
                # coord_.clamp_(-1 + 1e-6, 1 - 1e-6)

                coord_[:, :, 0] += r * delta_h
                coord_[:, :, 1] += c * delta_w
                coord_.clamp_(-1 + eps_shift, 1 - eps_shift)

                # (Pdb) coord_.size()
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
                q_feat = F.grid_sample(
                    feat, coord_.flip(-1).unsqueeze(1),
                    mode='nearest', align_corners=False)[:, :, 0, :] \
                    .permute(0, 2, 1)
                # (Pdb) q_feat.size() -- torch.Size([1, 30000, 576])

                q_coord = F.grid_sample(
                    feat_coord, coord_.flip(-1).unsqueeze(1),
                    mode='nearest', align_corners=False)[:, :, 0, :] \
                    .permute(0, 2, 1)
                # (Pdb) q_coord.size() -- torch.Size([1, 30000, 2])

                rel_coord = coord - q_coord
                rel_coord[:, :, 0] *= feat.shape[-2]
                rel_coord[:, :, 1] *= feat.shape[-1]
                # (Pdb) rel_coord.size() -- torch.Size([1, 30000, 2])
                # (Pdb) rel_coord.min(), rel_coord.max() -- (tensor(-0.9375, device='cuda:0'), tensor(1.9375, device='cuda:0'))
                inp = torch.cat([q_feat, rel_coord], dim=-1)

                rel_cell = cell.clone()
                rel_cell[:, :, 0] *= feat.shape[-2]
                rel_cell[:, :, 1] *= feat.shape[-1]
                # (Pdb) rel_cell.size() -- torch.Size([1, 30000, 2])
                # (Pdb) rel_cell.min(), rel_cell.max() -- (tensor(0.1250, device='cuda:0'), tensor(0.1250, device='cuda:0'))
                inp = torch.cat([inp, rel_cell], dim=-1)

                batch, chan = coord.shape[:2]
                # coord.size() -- torch.Size([1, 30000, 2])

                # MLP Forward ...
                pred = self.imnet(inp.view(batch * chan, -1)).view(batch, chan, -1)
                preds.append(pred)

                area = torch.abs(rel_coord[:, :, 0] * rel_coord[:, :, 1])
                areas.append(area + 1e-9)

        total_area = torch.stack(areas).sum(dim=0)

        t = areas[0]; areas[0] = areas[3]; areas[3] = t
        t = areas[1]; areas[1] = areas[2]; areas[2] = t

        ret = 0
        for pred, area in zip(preds, areas):
            ret = ret + pred * (area / total_area).unsqueeze(-1)
        return ret

    def forward(self, inp, coord, cell):
        pdb.set_trace()

        self.gen_feat(inp)
        return self.query_rgb(coord, cell)
