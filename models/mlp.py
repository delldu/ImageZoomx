import torch.nn as nn

from models import register
import pdb

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

    def forward(self, x):
        shape = x.shape[:-1]
        # x.size() -- torch.Size([65536, 3])
        x = self.layers(x.view(-1, x.shape[-1]))
        # (Pdb) pp shape -- torch.Size([65536])
        return x.view(shape[0], -1)
