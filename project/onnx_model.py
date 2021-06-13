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

    dummy_input = torch.randn(1, 3, 256, 256)
    device = model_device()
    dummy_input = dummy_input.to(device)

    onnx_file_name = "{}/image_zoomx.onnx".format(args.output)
    checkpoints = "models/ImageZoomx.pth"

    def export_onnx():
        """Export onnx model."""

        # 1. Create and load model.
        torch_model = get_model(checkpoints)
        # torch_model = torch_model.cuda()
        torch_model.eval()

        output = torch_model(dummy_input)

        pdb.set_trace()

        # 2. Model export
        print("Exporting onnx model to {}...".format(onnx_file_name))

        input_names = ["input"]
        output_names = ["output"]
        dynamic_axes = {'input': {2: "height", 3: 'width'},
                        'output': {2: "height", 3: 'width'}}

        torch.onnx.export(torch_model, dummy_input, onnx_file_name,
                          input_names=input_names,
                          output_names=output_names,
                          verbose=True,
                          opset_version=11,
                          keep_initializers_as_inputs=False,
                          export_params=True,
                          dynamic_axes=dynamic_axes)

        # 3. Optimize model
        # print('Checking model ...')
        # onnx_model = onnx.load(onnx_file_name)
        # onnx.checker.check_model(onnx_model)
        # https://github.com/onnx/optimizer

        # 4. Visual model
        # python -c "import netron; netron.start('output/image_zoomx.onnx')"

    def verify_onnx():
        """Verify onnx model."""

        sys.exit("Sorry, this function NOT work for grid_sampler, please use onnxservice to test.")

        torch_model = get_model(checkpoints)
        torch_model.eval()

        onnxruntime_engine = onnx_load(onnx_file_name)

        def to_numpy(tensor):
            return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

        with torch.no_grad():
            torch_output = torch_model(dummy_input)

        onnxruntime_inputs = {onnxruntime_engine.get_inputs()[0].name: to_numpy(dummy_input)}
        onnxruntime_outputs = onnxruntime_engine.run(None, onnxruntime_inputs)

        np.testing.assert_allclose(to_numpy(torch_output), onnxruntime_outputs[0], rtol=1e-03, atol=1e-03)
        print("Onnx model {} has been tested with ONNXRuntime, result sounds good !".format(onnx_file_name))

    #
    # /************************************************************************************
    # ***
    # ***    Flow Control
    # ***
    # ************************************************************************************/
    #

    if args.export:
        export_onnx()

    if args.verify:
        verify_onnx()
