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

import tvm
from tvm import relay, runtime, contrib

import onnx
import onnxruntime


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

    def build_encoder_model():
        onnx_file = "image_zoomx_encoder.onnx"
        print("Building {} on {} ...".format(onnx_file, target))

        onnx_model = onnx.load("output/{}".format(onnx_file))
        onnx_shape_dict = {"input": (1, 3, image_height, image_width)}

        devname = "cuda" if args.gpu else "cpu"
        onnx_so_path = "{}/{}_{}.so".format(args.output, devname, onnx_file)
        onnx_json_path = "{}/{}_{}.graph".format(args.output, devname, onnx_file)
        onnx_params_path = "{}/{}_{}.params".format(args.output, devname, onnx_file)

        mod, params = relay.frontend.from_onnx(
            onnx_model, shape=onnx_shape_dict, freeze_params=True
        )
        print(mod)

        with tvm.transform.PassContext(opt_level=3):
            graph, lib, params = relay.build(mod, target=target, params=params)

        lib.export_library(onnx_so_path)

        with open(onnx_json_path, "w") as json_file:
            json_file.write(graph)

        with open(onnx_params_path, "wb") as params_file:
            params_file.write(runtime.save_param_dict(params))

        print("Building {}/{} OK".format(args.output, onnx_model))

    def build_transform_model():
        onnx_file = "image_zoomx_transform.onnx"
        print("Building {} on {} ...".format(onnx_file, target))

        onnx_model = onnx.load("output/{}".format(onnx_file))
        onnx_shape_dict = {
            "feat": (1, 576, image_height, image_width),
            "grid": (1, 2, image_height, image_width),
            "sub_grid": (1, 1, 65536, 2),
            "sub_cell": (1, 1, 65536, 2),
        }

        devname = "cuda" if args.gpu else "cpu"
        onnx_so_path = "{}/{}_{}.so".format(args.output, devname, onnx_file)
        onnx_json_path = "{}/{}_{}.graph".format(args.output, devname, onnx_file)
        onnx_params_path = "{}/{}_{}.params".format(args.output, devname, onnx_file)

        mod, params = relay.frontend.from_onnx(
            onnx_model, shape=onnx_shape_dict, freeze_params=True
        )
        print(mod)

        with tvm.transform.PassContext(opt_level=3):
            graph, lib, params = relay.build(mod, target=target, params=params)

        lib.export_library(onnx_so_path)

        with open(onnx_json_path, "w") as json_file:
            json_file.write(graph)

        with open(onnx_params_path, "wb") as params_file:
            params_file.write(runtime.save_param_dict(params))

        print("Building {}/{} OK".format(args.output, onnx_model))

    def verify_encoder_model():
        onnx_file = "image_zoomx_encoder.onnx"
        print("Running {}/{} on {} ...".format(args.output, onnx_file, device))

        devname = "cuda" if args.gpu else "cpu"
        onnx_so_path = "{}/{}_{}.so".format(args.output, devname, onnx_file)
        onnx_json_path = "{}/{}_{}.graph".format(args.output, devname, onnx_file)
        onnx_params_path = "{}/{}_{}.params".format(args.output, devname, onnx_file)

        np_data = np.random.uniform(size=(1, 3, image_height, image_width)).astype(
            "float32"
        )
        nd_data = tvm.nd.array(np_data, device)

        # Load module
        graph = open(onnx_json_path).read()
        loaded_solib = runtime.load_module(onnx_so_path)
        loaded_params = bytearray(open(onnx_params_path, "rb").read())

        mod = contrib.graph_executor.create(graph, loaded_solib, device)
        mod.load_params(loaded_params)

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

        onnxruntime_engine = onnx_load("{}/{}".format(args.output, onnx_file))
        onnxruntime_inputs = {onnxruntime_engine.get_inputs()[0].name: np_data}
        onnxruntime_outputs = onnxruntime_engine.run(None, onnxruntime_inputs)

        np.testing.assert_allclose(
            output_data.numpy(), onnxruntime_outputs[0], rtol=1e-03, atol=1e-03
        )
        print("Running {}/{} OK.".format(args.output, onnx_file))

    def verify_transform_model():
        onnx_file = "image_zoomx_transform.onnx"
        print("Running {}/{} on {} ...".format(args.output, onnx_file, device))

        devname = "cuda" if args.gpu else "cpu"
        onnx_so_path = "{}/{}_{}.so".format(args.output, devname, onnx_file)
        onnx_json_path = "{}/{}_{}.graph".format(args.output, devname, onnx_file)
        onnx_params_path = "{}/{}_{}.params".format(args.output, devname, onnx_file)

        np_feat_data = np.random.uniform(
            size=(1, 576, image_height, image_width)
        ).astype("float32")
        nd_feat_data = tvm.nd.array(np_feat_data, device)

        np_grid_data = np.random.uniform(size=(1, 2, image_height, image_width)).astype(
            "float32"
        )
        nd_grid_data = tvm.nd.array(np_grid_data, device)

        np_sub_grid_data = np.random.uniform(size=(1, 1, 65536, 2)).astype("float32")
        nd_sub_grid_data = tvm.nd.array(np_sub_grid_data, device)

        np_sub_cell_data = np.random.uniform(size=(1, 1, 65536, 2)).astype("float32")
        nd_sub_cell_data = tvm.nd.array(np_sub_cell_data, device)

        # Load module
        graph = open(onnx_json_path).read()
        loaded_solib = runtime.load_module(onnx_so_path)
        loaded_params = bytearray(open(onnx_params_path, "rb").read())

        mod = contrib.graph_executor.create(graph, loaded_solib, device)
        mod.load_params(loaded_params)

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

        onnxruntime_engine = onnx_load("{}/{}".format(args.output, onnx_file))
        onnxruntime_inputs = {
            onnxruntime_engine.get_inputs()[0].name: np_feat_data,
            onnxruntime_engine.get_inputs()[1].name: np_grid_data,
            onnxruntime_engine.get_inputs()[2].name: np_sub_grid_data,
            onnxruntime_engine.get_inputs()[3].name: np_sub_cell_data,
        }
        onnxruntime_outputs = onnxruntime_engine.run(None, onnxruntime_inputs)

        np.testing.assert_allclose(
            output_data.numpy(), onnxruntime_outputs[0], rtol=1e-03, atol=1e-03
        )
        print("Running {}/{} OK.".format(args.output, onnx_file))

    if args.build:
        # build_encoder_model()
        build_transform_model()

    if args.verify:
        # verify_encoder_model()
        verify_transform_model()
