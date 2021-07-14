"""Model Build."""  # coding=utf-8
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

import pdb

checkpoints = "models/ImageZoomx.pth"


input_height = 512
input_width = 512

model_input = torch.randn(1, 3, input_height, input_width)

trans_feat = torch.randn(1, 576, input_height, input_width)
trans_feat_grid = torch.randn(1, 2, input_height, input_width)
trans_sub_grid = torch.randn(1, 1, 65536, 2)
trans_sub_cell = torch.randn(1, 1, 65536, 2)

input_shapes = [
    ("input", ([1, 3, input_height, input_width], "float32")),
    ("output_size", ([2], "float32")),
]
encoder_input_shapes = [("input", ([1, 3, input_height, input_width], "float32"))]

trans_input_shapes = [
    ("feat", ([1, 576, input_height, input_width], "float32")),
    ("feat_grid", ([1, 2, input_height, input_width], "float32")),
    ("sub_grid", ([1, 1, 65536, 2], "float32")),
    ("sub_cell", ([1, 1, 65536, 2], "float32")),
]

model = get_model(checkpoints)

def build_encoder_script_model():
    script_model = torch.jit.script(model.encoder)
    print(script_model.graph)
    print("model.encoder.script code---------------------")
    print(script_model.code)
    mod, params = relay.frontend.from_pytorch(script_model, encoder_input_shapes)

def build_encoder_traced_model():
    traced_model = torch.jit.trace(model.encoder, model_input)
    print(traced_model.graph)
    print("model.encoder.traced code---------------------")
    print(traced_model.code)
    mod, params = relay.frontend.from_pytorch(traced_model, encoder_input_shapes)
    print("Building encoder model OK ...")
    pdb.set_trace()

def build_transform_script_model():
    script_model = torch.jit.script(model.imnet)
    print(script_model.graph)
    print("model.imnet.script code---------------------")
    print(script_model.code)
    mod, params = relay.frontend.from_pytorch(script_model, trans_input_shapes)
    pdb.set_trace()

def build_transform_traced_model():
    traced_model = torch.jit.trace(model.imnet, (trans_feat, trans_feat_grid, trans_sub_grid, trans_sub_cell))
    print(traced_model.graph)
    print("model.imnet.traced code---------------------")
    print(traced_model.code)
    mod, params = relay.frontend.from_pytorch(traced_model, trans_input_shapes)
    print("Building transform model OK ...")
    pdb.set_trace()

def build_whole_script_model():
    script_model = torch.jit.script(model)
    print(script_model.graph)
    print("model script code---------------------")
    print(script_model.code)
    mod, params = relay.frontend.from_pytorch(script_model, input_shapes)
    print("Building whole model OK ...")
    pdb.set_trace()

def build_whole_traced_model():
    traced_model = torch.jit.trace(model, (model_input, torch.Tensor([1024.0, 1024.0])))
    print(traced_model.graph)
    print("model traced code---------------------")
    print(traced_model.code)
    mod, params = relay.frontend.from_pytorch(traced_model, input_shapes)
    print("Building whole model OK ...")
    # ['aten::append', 'aten::grid_sampler', 'aten::flip', 'aten::im2col']
    pdb.set_trace()

def build_traced_model():
    build_encoder_traced_model()
    build_transform_traced_model()
    build_whole_traced_model()

def build_script_model():
    build_encoder_script_model()
    build_transform_script_model()
    build_whole_script_model()

build_traced_model()
# build_script_model()
