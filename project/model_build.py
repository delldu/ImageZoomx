"""Model Build."""# coding=utf-8
#
# /************************************************************************************
# ***
# ***    Copyright Dell 2021, All Rights Reserved.
# ***
# ***    File Author: Dell, 2021年 05月 27日 星期四 20:27:47 CST
# ***
# ************************************************************************************/
#

import torch
import torch.nn as nn

import tvm
from tvm import relay
import tvm.testing
from model import get_model, model_device, model_setenv


checkpoints = "models/ImageZoomx.pth"


input_height = 512
input_width = 512

model_input = torch.randn(1, 3, input_height, input_width)

trans_feat = torch.randn(1, 576, input_height, input_width)
trans_feat_grid = torch.randn(1, 2, input_height, input_width)
trans_sub_grid = torch.randn(1, 1, 65536, 2)
trans_sub_cell = torch.randn(1, 1, 65536, 2)


model = get_model(checkpoints)
model.encoder = torch.jit.trace(model.encoder, model_input)
model.imnet = torch.jit.trace(model.imnet, (trans_feat, trans_feat_grid, trans_sub_grid, trans_sub_cell), check_trace=False)
script_model = torch.jit.script(model)

# script_model = torch.jit.trace(model, (model_input, torch.Tensor([1024.0, 1024.0])))


input_shapes = [
    ('input', ([1, 3, input_height, input_width], 'float32')),
    ('output_size', ([2], 'float32'))
]
encoder_input_shapes = [
    ('input', ([1, 3, input_height, input_width], 'float32'))
]

trans_shapes = [
    ('feat', ([1, 576, input_height, input_width], 'float32')),
    ('feat_grid', ([1, 2, input_height, input_width], 'float32')),
    ('sub_grid', ([1, 1, 65536, 2], 'float32')),
    ('sub_cell', ([1, 1, 65536, 2], 'float32'))
]



mod, params = relay.frontend.from_pytorch(script_model, input_shapes)

#  ['aten::append', 'aten::grid_sampler', 'aten::flip', 'aten::im2col']
pdb.set_trace()

