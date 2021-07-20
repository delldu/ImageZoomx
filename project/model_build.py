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

import argparse
import os
import pdb

import numpy as np

import torch

import tvm
from tvm import relay, runtime, contrib

from model import get_model


def save_tvm_model(relay_mod, relay_params, target, filename):
    # Create ouput file names
    file_name, _ = os.path.splitext(filename)
    so_filename = file_name + ".so"
    json_filename = file_name + ".json"
    params_filename = file_name + ".params"

    with tvm.transform.PassContext(opt_level=3):
        graph, lib, params = relay.build(relay_mod, target=target, params=relay_params)

    lib.export_library(so_filename)

    with open(json_filename, "w") as f:
        f.write(graph)

    with open(params_filename, "wb") as f:
        f.write(runtime.save_param_dict(params))

    print("Building {} OK".format(file_name))


def load_tvm_model(filename, device):
    # Create input file names
    file_name, _ = os.path.splitext(filename)
    so_filename = file_name + ".so"
    json_filename = file_name + ".json"
    params_filename = file_name + ".params"

    graph = open(json_filename).read()
    loaded_solib = runtime.load_module(so_filename)
    loaded_params = bytearray(open(params_filename, "rb").read())

    mod = contrib.graph_executor.create(graph, loaded_solib, device)
    mod.load_params(loaded_params)

    return mod


if __name__ == "__main__":
    """TVM Onnx tools ..."""

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("-b", "--build", help="build tvm model", action="store_true")
    parser.add_argument("-v", "--verify", help="verify tvm model", action="store_true")
    parser.add_argument("-g", "--gpu", help="use gpu", action="store_true")
    parser.add_argument(
        "-o", "--output", type=str, default="output", help="output folder"
    )

    args = parser.parse_args()

    if not os.path.exists(args.output):
        os.makedirs(args.output)

    # /************************************************************************************
    # ***
    # ***    MS: Define Global Names
    # ***
    # ************************************************************************************/
    target = tvm.target.Target("cuda" if args.gpu else "llvm", host="llvm")
    device = tvm.device(str(target), 0)
    image_height = 256
    image_width = 256
    checkpoints = "models/ImageZoomx.pth"
    prefix = "cuda" if args.gpu else "cpu"

    encoder_input_shape = (1, 3, image_height, image_width)
    output_encoder_so_filename = "{}/{}_image_zoomx_encoder.so".format(
        args.output, prefix
    )
    output_encoder_traced_filename = "{}/{}_image_zoomx_encoder.pt".format(
        args.output, prefix
    )

    # for transform
    feat_input_shape = (1, 576, image_height, image_width)
    grid_input_shape = (1, 2, image_height, image_width)
    sub_grid_input_shape = (1, 1, 65536, 2)
    sub_cell_input_shape = (1, 1, 65536, 2)
    output_transform_so_filename = "{}/{}_image_zoomx_transform.so".format(
        args.output, prefix
    )
    output_transform_traced_filename = "{}/{}_image_zoomx_transform.pt".format(
        args.output, prefix
    )

    def build_encoder_model():
        print("Building model on {} ...".format(target))

        encoder_input_data = torch.randn(encoder_input_shape)

        model = get_model(checkpoints).encoder
        model = model.eval()

        traced_model = torch.jit.trace(model, encoder_input_data)
        traced_model.save(output_encoder_traced_filename)

        print(traced_model.graph)
        mod, params = relay.frontend.from_pytorch(
            traced_model, [("input", encoder_input_shape)]
        )
        save_tvm_model(mod, params, target, output_encoder_so_filename)

    def build_transform_model():
        print("Building model on {} ...".format(target))

        feat_input_data = torch.randn(feat_input_shape)
        grid_input_data = torch.randn(grid_input_shape)
        sub_grid_input_data = torch.randn(sub_grid_input_shape)
        sub_cell_input_data = torch.randn(sub_cell_input_shape)

        model = get_model(checkpoints).imnet
        model = model.eval()

        traced_model = torch.jit.trace(
            model,
            (
                feat_input_data,
                grid_input_data,
                sub_grid_input_data,
                sub_cell_input_data,
            ),
        )
        traced_model.save(output_transform_traced_filename)

        print(traced_model.graph)
        mod, params = relay.frontend.from_pytorch(
            traced_model,
            [
                ("feat", feat_input_shape),
                ("grid", grid_input_shape),
                ("sub_grid", sub_grid_input_shape),
                ("sub_cell", sub_cell_input_shape),
            ],
        )
        save_tvm_model(mod, params, target, output_transform_so_filename)

    def verify_encoder_model():
        print("Running model on {} ...".format(device))

        input_data = torch.randn(encoder_input_shape)
        nd_data = tvm.nd.array(input_data.numpy(), device)

        # Load module
        mod = load_tvm_model(output_encoder_so_filename, device)

        # TVM Run
        mod.set_input("input", nd_data)
        mod.run()
        output_data = mod.get_output(0)
        print(output_data)

        print("Evaluating ...")
        ftimer = mod.module.time_evaluator("run", device, number=2, repeat=10)
        prof_res = np.array(ftimer().results) * 1000  # multiply 1000 for ms
        print(
            "Mean running time: %.2f ms (stdv: %.2f ms)"
            % (np.mean(prof_res), np.std(prof_res))
        )

        traced_model = torch.jit.load(output_encoder_traced_filename)
        traced_model = traced_model.eval()
        with torch.no_grad():
            traced_output = traced_model(input_data)

        np.testing.assert_allclose(
            output_data.numpy(), traced_output.numpy(), rtol=1e-03, atol=1e-03
        )
        print("Running model OK.")

    def verify_transform_model():
        print("Running model on {} ...".format(device))

        feat_data = torch.randn(feat_input_shape)
        nd_feat_data = tvm.nd.array(feat_data.numpy(), device)

        grid_data = torch.randn(grid_input_shape)
        nd_grid_data = tvm.nd.array(grid_data.numpy(), device)

        sub_grid_data = torch.randn(sub_grid_input_shape)
        nd_sub_grid_data = tvm.nd.array(sub_grid_data.numpy(), device)

        sub_cell_data = torch.randn(sub_cell_input_shape)
        nd_sub_cell_data = tvm.nd.array(sub_cell_data.numpy(), device)

        # Load module
        mod = load_tvm_model(output_transform_so_filename, device)

        # TVM Run
        mod.set_input("feat", nd_feat_data)
        mod.set_input("grid", nd_grid_data)
        mod.set_input("sub_grid", nd_sub_grid_data)
        mod.set_input("sub_cell", nd_sub_cell_data)

        mod.run()
        output_data = mod.get_output(0)
        print(output_data)

        print("Evaluating ...")
        ftimer = mod.module.time_evaluator("run", device, number=2, repeat=10)
        prof_res = np.array(ftimer().results) * 1000  # multiply 1000 for ms
        print(
            "Mean running time: %.2f ms (stdv: %.2f ms)"
            % (np.mean(prof_res), np.std(prof_res))
        )

        traced_model = torch.jit.load(output_transform_traced_filename)
        traced_model = traced_model.eval()
        with torch.no_grad():
            traced_output = traced_model(
                feat_data, grid_data, sub_grid_data, sub_cell_data
            )

        np.testing.assert_allclose(
            output_data.numpy(), traced_output.numpy(), rtol=1e-03, atol=1e-03
        )
        print("Running model OK.")

    if args.build:
        build_encoder_model()
        build_transform_model()

    if args.verify:
        verify_encoder_model()
        verify_transform_model()
