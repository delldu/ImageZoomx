import argparse
import os
from PIL import Image

import torch
from torchvision import transforms

import models
from utils import make_coord
from test import batched_predict

import pdb

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', default='input.png')
    parser.add_argument('--model')
    parser.add_argument('--resolution')
    parser.add_argument('--output', default='output.png')
    parser.add_argument('--gpu', default='0')
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    img = transforms.ToTensor()(Image.open(args.input).convert('RGB'))

    # pdb.set_trace()
    # x = torch.load(args.model)
    # (Pdb) x.keys() -- dict_keys(['model', 'optimizer', 'epoch'])
    # x['model']['sd'].keys()
    # odict_keys(['encoder.sub_mean.weight', 'encoder.sub_mean.bias', 
    #     'encoder.add_mean.weight', 'encoder.add_mean.bias', 'encoder.head.0.weight', 
    #     ...

    model = models.make(torch.load(args.model)['model'], load_sd=True).cuda()


    h, w = list(map(int, args.resolution.split(',')))
    h *= 4
    w *= 4
    
    coord = make_coord((h, w)).cuda()
    cell = torch.ones_like(coord)
    cell[:, 0] *= 2 / h
    cell[:, 1] *= 2 / w

    # pdb.set_trace()
    # coord.size() -- [h * w, 2]

    # pred = batched_predict(model, ((img - 0.5) / 0.5).cuda().unsqueeze(0),
    #     coord.unsqueeze(0), cell.unsqueeze(0), bsize=30000)[0]
    # pred = (pred * 0.5 + 0.5).clamp(0, 1).view(h, w, 3).permute(2, 0, 1).cpu()


    pred = batched_predict(model, img.cuda().unsqueeze(0),
        coord.unsqueeze(0), cell.unsqueeze(0), bsize=30000)[0]
    pred = pred.clamp(0, 1).view(h, w, 3).permute(2, 0, 1).cpu()

    transforms.ToPILImage()(pred).save(args.output)
