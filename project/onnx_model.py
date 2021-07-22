"""Onnx Model Tools."""  # coding=utf-8
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

# from torch.onnx.symbolic_helper import parse_args
# from torch.onnx.symbolic_registry import register_op

# @parse_args('v', 'v', 'i', 'i', 'i')
# def grid_sampler(g, input, grid, interpolation_mode, padding_mode, align_corners=False):
#     '''
#     torch.nn.functional.grid_sample(input, grid, mode='bilinear', padding_mode='zeros', align_corners=None)
#     Need convert interpolation_mode, padding_mode ? NO for simpler at now !!!
#     '''
#     return g.op('onnxservice::grid_sampler', input, grid,
#         interpolation_mode_i=interpolation_mode,
#         padding_mode_i=padding_mode,
#         align_corners_i=align_corners)

# register_op('grid_sampler', grid_sampler, '', 11)


def to_numpy(tensor):
    return (
        tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()
    )


def onnx_load(onnx_file):
    session_options = onnxruntime.SessionOptions()
    # session_options.log_severity_level = 0

    # Set graph optimization level
    session_options.graph_optimization_level = (
        onnxruntime.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
    )

    onnx_model = onnxruntime.InferenceSession(onnx_file, session_options)
    # onnx_model.set_providers(['CUDAExecutionProvider'])
    print(
        "Onnx Model Engine: ",
        onnx_model.get_providers(),
        "Device: ",
        onnxruntime.get_device(),
    )

    return onnx_model


if __name__ == "__main__":
    """Onnx tools ..."""

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("-e", "--export", help="export onnx model", action="store_true")
    parser.add_argument("-v", "--verify", help="verify onnx model", action="store_true")
    parser.add_argument(
        "-o", "--output", type=str, default="output", help="output folder"
    )

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

    image_height = 256
    image_width = 256

    dummy_encoder_input = torch.randn(1, 3, image_height, image_width)

    dummy_transform_feat = torch.randn(1, 576, image_height, image_width)
    dummy_transform_feat_grid = torch.randn(1, 2, image_height, image_width)
    dummy_transform_sub_grid = torch.randn(1, 2, 65536, 1)
    dummy_transform_sub_cell = torch.randn(1, 2, 65536, 1)

    def export_encoder_onnx():
        # 1. Create and load model.
        model = get_model(checkpoints).encoder
        model.eval()

        # 2. Model export
        print("Exporting onnx model to {}...".format(encoder_onnx_file_name))

        input_names = ["input"]
        output_names = ["output"]
        dynamic_axes = {
            "input": {2: "height", 3: "width"},
            "output": {2: "height", 3: "width"},
        }
        torch.onnx.export(
            model,
            dummy_encoder_input,
            encoder_onnx_file_name,
            input_names=input_names,
            output_names=output_names,
            verbose=True,
            opset_version=11,
            keep_initializers_as_inputs=False,
            dynamic_axes=dynamic_axes,
            export_params=True,
        )

        # 3. Optimize model
        # https://github.com/onnx/optimizer
        print("Checking model {}...".format(encoder_onnx_file_name), end=" ")
        onnx_model = onnx.load(encoder_onnx_file_name)
        onnx.checker.check_model(onnx_model)
        print("OK")

        # 4. Visual model
        # python -c "import netron; netron.start('output/image_zoomx_encoder.onnx')"

    def export_transform_onnx():
        # 1. Create and load model.
        model = get_model(checkpoints).imnet
        model.eval()

        # 2. Model export
        print("Exporting onnx model to {}...".format(transform_onnx_file_name))

        input_names = ["feat", "grid", "sub_grid", "sub_cell"]
        output_names = ["output"]
        dynamic_axes = {
            "feat": {2: "height", 3: "width"},
            "grid": {2: "height", 3: "width"},
            "output": {2: "height", 3: "width"},
        }
        torch.onnx.export(
            model,
            (
                dummy_transform_feat,
                dummy_transform_feat_grid,
                dummy_transform_sub_grid,
                dummy_transform_sub_cell,
            ),
            transform_onnx_file_name,
            input_names=input_names,
            output_names=output_names,
            verbose=True,
            opset_version=11,
            keep_initializers_as_inputs=False,
            dynamic_axes=dynamic_axes,
            export_params=True,
        )

        # 3. Optimize model
        # https://github.com/onnx/optimizer
        print("Checking model {} ...".format(transform_onnx_file_name), end=" ")
        onnx_model = onnx.load(transform_onnx_file_name)
        onnx.checker.check_model(onnx_model)
        print("OK")

        # 4. Visual model
        # python -c "import netron; netron.start('output/image_zoomx_transform.onnx')"

    def verify_encoder_onnx():
        model = get_model(checkpoints).encoder
        model.eval()

        onnxruntime_engine = onnx_load(encoder_onnx_file_name)

        with torch.no_grad():
            torch_output = model(dummy_encoder_input)

        onnxruntime_inputs = {
            onnxruntime_engine.get_inputs()[0].name: to_numpy(dummy_encoder_input)
        }
        onnxruntime_outputs = onnxruntime_engine.run(None, onnxruntime_inputs)

        np.testing.assert_allclose(
            to_numpy(torch_output), onnxruntime_outputs[0], rtol=1e-03, atol=1e-03
        )
        print("Test model {} with ONNXRuntime OK !".format(encoder_onnx_file_name))

    def verify_transform_onnx():
        model = get_model(checkpoints).imnet
        model.eval()

        onnxruntime_engine = onnx_load(transform_onnx_file_name)

        with torch.no_grad():
            torch_output = model(
                dummy_transform_feat, dummy_transform_sub_grid, dummy_transform_sub_cell
            )

        onnxruntime_inputs = {
            onnxruntime_engine.get_inputs()[0].name: to_numpy(dummy_encoder_feat),
            onnxruntime_engine.get_inputs()[1].name: to_numpy(dummy_encoder_grid),
            onnxruntime_engine.get_inputs()[2].name: to_numpy(dummy_encoder_cell),
        }
        onnxruntime_outputs = onnxruntime_engine.run(None, onnxruntime_inputs)

        np.testing.assert_allclose(
            to_numpy(torch_output), onnxruntime_outputs[0], rtol=1e-03, atol=1e-03
        )
        print("Test model {} with ONNXRuntime OK !".format(encoder_onnx_file_name))

    # /************************************************************************************
    # ***
    # ***    Flow Control
    # ***
    # ************************************************************************************/

    if args.export:
        export_encoder_onnx()
        export_transform_onnx()

    if args.verify:
        verify_encoder_onnx()
        verify_transform_onnx()
        # verify_onnx()
        # For onnx does not support grid_sampler, please verify it in onnxruntime service
        print("model does not been tested for onnx not support grid_sampler !")
