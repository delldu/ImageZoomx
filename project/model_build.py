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

import numpy as np

import tvm
from tvm import relay
import tvm.testing

from model import get_model, model_device, model_setenv

import argparse
import os
import pdb

if __name__ == '__main__':
    """TVM Onnx tools ..."""

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-b', '--build', help="build tvm model", action='store_true')
    parser.add_argument('-v', '--verify', help="verify tvm model", action='store_true')
    parser.add_argument('-g', '--gpu', help="use gpu", action='store_true')
    parser.add_argument('-o', '--output', type=str, default="models", help="output folder")

    args = parser.parse_args()

    if not os.path.exists(args.output):
        os.makedirs(args.output)

    # /************************************************************************************
    # ***
    # ***    MS: Define Global Names
    # ***
    # ************************************************************************************/
    if args.gpu:
        target = tvm.target.Target("cuda", host='llvm')
    else:
        target = tvm.target.Target("llvm", host='llvm')
    device = tvm.device(str(target), 0)        

    checkpoints = "models/ImageZoomx.pth"

    input_height = 256
    input_width = 256
    input_shape = [1, 3, input_height, input_width]
    model_input = torch.randn(1, 3, input_height, input_width)
    input_shapes = [
        ("input", ([1, 3, input_height, input_width], "float32")),
        ("output_size", ([2], "float32")),
    ]

    so_path = "{}/image_zoomx.so".format(args.output)
    json_path = "{}/image_zoomx.json".format(args.output)
    params_path = "{}/image_zoomx.params".format(args.output)

    model = get_model(checkpoints)

    # def build_script_model():
    #     script_model = torch.jit.script(model)
    #     print(script_model.graph)
    #     print("model script code---------------------")
    #     print(script_model.code)
    #     mod, params = relay.frontend.from_pytorch(script_model, input_shapes)
    #     print("Building whole model OK ...")

    def build_traced_model():
        traced_model = torch.jit.trace(model, (model_input, torch.Tensor([1024.0, 1024.0])))
        print(traced_model.graph)
        print("model traced code---------------------")
        print(traced_model.code)
        mod, params = relay.frontend.from_pytorch(traced_model, input_shapes)

        with relay.build_config(opt_level=3):
            graph, lib, params = relay.build_module.build(mod, target, params=params)

        # Output TVM model
        lib.export_library(so_path)
        with open(json_path, 'w') as fo:
            fo.write(graph)
        with open(params_path, 'wb') as fo:
            fo.write(relay.save_param_dict(params))

        print("Building model OK ...")
        # ['aten::append', 'aten::grid_sampler', 'aten::flip', 'aten::im2col']


    def verify_traced_model():
        loaded_json = open(json_path).read()
        loaded_solib = tvm.runtime.load_module(so_path)
        loaded_params = bytearray(open(params_path, "rb").read())

        module =  tvm.contrib.graph_executor.create(loaded_json, loaded_solib, device)
        module.load_params(loaded_params)

        # Predict
        # /************************************************************************************
        # ***
        # ***    MS: Define Input Data
        # ***
        # ************************************************************************************/        
        input = tvm.nd.array((np.random.uniform(size=input_shape).astype("float32")), device)
        output_size = tvm.nd.array((np.random.uniform(size=(2,)).astype("float32")), device)

        module.set_input("input", input)
        # module.set_input("output_size", output_size)
        module.run()
        output = module.get_output(0)
        print(output)


        print("Evaluating ...")
        ftimer = module.module.time_evaluator("run", device, number=1, repeat=5)
        prof_res = np.array(ftimer().results) * 1000  # multiply 1000 for converting to millisecond
        print(
            "%-20s %-19s (%s)" % (so_path, "%.2f ms" % np.mean(prof_res), "%.2f ms" % np.std(prof_res))
        )


    if args.build:
        build_traced_model()

    if args.verify:
        verify_traced_model()
