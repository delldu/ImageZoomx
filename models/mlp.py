import torch
import torch.nn as nn
import torch.nn.functional as F

from models import register
import pdb

def make_coord(shape, flatten=True):
    """ Make coordinates at grid centers.
    """
    coord_seqs = []
    for i, n in enumerate(shape):
        v0, v1 = -1, 1
        r = (v1 - v0) / (2 * n)
        seq = v0 + r + (2 * r) * torch.arange(n).float()
        coord_seqs.append(seq)

    ret = torch.stack(torch.meshgrid(*coord_seqs), dim=-1)
    
    if flatten:
        ret = ret.view(-1, ret.shape[-1])

    return ret


@register('mlp')
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
        shape = x.shape[:-1]
        # x.size() -- torch.Size([65536, 3])
        x = self.layers(x.view(-1, x.shape[-1]))
        # (Pdb) pp shape -- torch.Size([65536])
        return x.view(shape[0], -1)


    def forward(self, feat, s_grid, s_cell):
        # (Pdb) pp s_grid.size(), s_cell.size()
        # (torch.Size([1, 30000, 2]), torch.Size([1, 30000, 2]))

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

                pred = self.simple_forward(input).view(batch, chan, -1)
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
