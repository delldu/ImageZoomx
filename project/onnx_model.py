"""Onnx Model Tools."""# coding=utf-8
#
# /************************************************************************************
# ***
# ***    Copyright Dell 2021, All Rights Reserved.
# ***
# ***    File Author: Dell, 2021年 05月 27日 星期四 20:27:47 CST
# ***
# ************************************************************************************/
#

import argparse
import os
import pdb  # For debug
import time
import sys

import numpy as np
import onnx
import onnxruntime
import torch
import torchvision.transforms as transforms
from PIL import Image

#
# /************************************************************************************
# ***
# ***    MS: Import Model Method
# ***
# ************************************************************************************/
#
from model import get_model, model_device, model_setenv

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


def onnx_load(onnx_file):
    session_options = onnxruntime.SessionOptions()
    # session_options.log_severity_level = 0

    # Set graph optimization level
    session_options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_EXTENDED

    onnx_model = onnxruntime.InferenceSession(onnx_file, session_options)
    # onnx_model.set_providers(['CUDAExecutionProvider'])
    print("Onnx Model Engine: ", onnx_model.get_providers(),
          "Device: ", onnxruntime.get_device())

    return onnx_model


def onnx_forward(onnx_model, input):
    def to_numpy(tensor):
        return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

    onnxruntime_inputs = {onnx_model.get_inputs()[0].name: to_numpy(input)}
    onnxruntime_outputs = onnx_model.run(None, onnxruntime_inputs)
    return torch.from_numpy(onnxruntime_outputs[0])


if __name__ == '__main__':
    """Onnx tools ..."""

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-e', '--export', help="export onnx model", action='store_true')
    parser.add_argument('-v', '--verify', help="verify onnx model", action='store_true')
    parser.add_argument('-o', '--output', type=str, default="output", help="output folder")

    args = parser.parse_args()

    if not os.path.exists(args.output):
        os.makedirs(args.output)

    model_setenv()

    #
    # /************************************************************************************
    # ***
    # ***    MS: Define Global Names
    # ***
    # ************************************************************************************/
    #

    checkpoints = "models/ImageZoomx.pth"
    encoder_onnx_file_name = "{}/image_zoomx_encoder.onnx".format(args.output)
    transform_onnx_file_name = "{}/image_zoomx_transform.onnx".format(args.output)

    dummy_encoder_input = torch.randn(1, 3, 256, 256)
    dummy_output_size = torch.IntTensor([1024, 1024])

    dummy_transform_feat = torch.randn(1, 576, 96, 128)
    dummy_transform_grid = torch.randn(1, 1, 65536, 2)
    dummy_transform_cell = torch.randn(1, 1, 65536, 2)

    def export_encoder_onnx():
        """Export onnx model."""

        # 1. Create and load model.
        torch_model = get_model(checkpoints)
        torch_model.eval()
        script_model = torch.jit.script(torch_model)
        # traced_model = torch.jit.trace(script_model, (dummy_encoder_input, dummy_output_size))
        pdb.set_trace()

        with torch.no_grad():
            torch_output = torch_model(dummy_encoder_input, dummy_output_size)

        # 2. Model export
        print("Exporting onnx model to {}...".format(encoder_onnx_file_name))

        input_names = ["input", "output_size"]
        output_names = ["output"]
        dynamic_axes = {'input': {2: "height", 3: 'width'},
                        'output': {2: "height", 3: 'width'}}
        torch.onnx.export(script_model, (dummy_encoder_input, dummy_output_size), encoder_onnx_file_name,
                          input_names=input_names,
                          output_names=output_names,
                          verbose=True,
                          opset_version=11,
                          keep_initializers_as_inputs=False,
                          export_params=True,
                          dynamic_axes=dynamic_axes,
                          example_outputs=torch_output)

        # 3. Visual model
        # python -c "import netron; netron.start('output/image_zoomx_encoder.onnx')"


    def export_transform_onnx():
        """Export transform onnx model."""

        # 1. Create and load model.
        torch_model = get_model(checkpoints).imnet
        torch_model.eval()
        script_model = torch.jit.script(torch_model)
        pdb.set_trace()

        # 2. Model export
        print("Exporting onnx model to {}...".format(transform_onnx_file_name))

        input_names = ["feat", "grid", "cell"]
        output_names = ["output"]
        dynamic_axes = {'grid': {2: "batch_size"},
                        'cell': {2: "batch_size"},
                        'output': {1: "batch_size"}}
        torch.onnx.export(torch_model, (dummy_transform_feat, dummy_transform_grid, dummy_transform_cell),
                          transform_onnx_file_name,
                          input_names=input_names,
                          output_names=output_names,
                          verbose=True,
                          opset_version=11,
                          keep_initializers_as_inputs=False,
                          export_params=True,
                          dynamic_axes=dynamic_axes)

        # 3. Visual model
        # python -c "import netron; netron.start('output/image_zoomx_transform.onnx')"


    def verify_encoder_onnx():
        """Verify encoder onnx model."""

        torch_model = get_model(checkpoints).encoder
        torch_model.eval()

        onnxruntime_engine = onnx_load(encoder_onnx_file_name)

        def to_numpy(tensor):
            return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

        with torch.no_grad():
            torch_output = torch_model(dummy_encoder_input)

        onnxruntime_inputs = {onnxruntime_engine.get_inputs()[0].name: to_numpy(dummy_encoder_input)}
        onnxruntime_outputs = onnxruntime_engine.run(None, onnxruntime_inputs)

        np.testing.assert_allclose(to_numpy(torch_output), onnxruntime_outputs[0], rtol=1e-03, atol=1e-03)
        print("Onnx model {} has been tested with ONNXRuntime, result sounds good !".format(encoder_onnx_file_name))


    def verify_transform_onnx():
        """Verify transform onnx model."""

        torch_model = get_model(checkpoints).imnet
        torch_model.eval()

        onnxruntime_engine = onnx_load(transform_onnx_file_name)

        def to_numpy(tensor):
            return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

        with torch.no_grad():
            torch_output = torch_model(dummy_transform_feat, dummy_transform_grid, dummy_transform_cell)

        onnxruntime_inputs = {onnxruntime_engine.get_inputs()[0].name: to_numpy(dummy_encoder_feat),
            onnxruntime_engine.get_inputs()[1].name: to_numpy(dummy_encoder_grid),
            onnxruntime_engine.get_inputs()[2].name: to_numpy(dummy_encoder_cell)}
        onnxruntime_outputs = onnxruntime_engine.run(None, onnxruntime_inputs)

        np.testing.assert_allclose(to_numpy(torch_output), onnxruntime_outputs[0], rtol=1e-03, atol=1e-03)
        print("Onnx model {} has been tested with ONNXRuntime, result sounds good !".format(encoder_onnx_file_name))


    #
    # /************************************************************************************
    # ***
    # ***    Flow Control
    # ***
    # ************************************************************************************/
    #

    if args.export:
        export_encoder_onnx()
        export_transform_onnx()

    if args.verify:
        verify_encoder_onnx()

        # For onnx does not support grid_sampler, please verify it in onnxruntime service
        print("Transform does not been tested here for onnx not support grid_sampler at now")
        # verify_transform_onnx()
