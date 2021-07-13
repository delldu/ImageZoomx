import torch
import torch.nn as nn

import tvm
from tvm import relay
import tvm.testing
from typing import List, Tuple
import torch.nn.functional as F

import pdb

def verify_model_with_vm(script_model, input_shapes, input_data=None, targets=['llvm', 'cuda', 'llvm -device=arm_cpu']):
    print("Verify model {} with vm ...".format(script_model))

    # input_shapes = [(1, 3, 224, 224)]
    # targets = ['cuda', 'llvm -device=arm_cpu', 'llvm']

    if input_data is None:
        input_data = [torch.randn(shape, dtype=torch.float32) for shape in input_shapes]

    input_names = ["i{}".format(idx) for idx, ish in enumerate(input_shapes)] #input_names -- ['i0']
    input_dtypes = ['float32'] * len(input_names)
    input_shapes = list(zip(input_names, list(zip(input_shapes, input_dtypes))))
    # input_shapes -- [('i0', ((10, 20), 'float32'))]

    # len(input_data), type(input_data[0]), input_data[0].size()
    # -- (1, <class 'torch.Tensor'>, torch.Size([10, 20]))
    # input_shapes = [('i0', ((10, 20), 'float32'))]

    # Compile via VM
    mod, params = relay.frontend.from_pytorch(script_model, input_shapes)
    # (Pdb) type(mod), type(params) -- (<class 'tvm.ir.module.IRModule'>, <class 'dict'>)

    # targets -- ['cuda', 'llvm -device=arm_cpu', 'llvm']
    for tgt in targets:
        print("Running {} on target ... ".format(script_model.original_name), tgt)
        print("input_data:", input_data)

        dev = tvm.device(tgt, 0)

        executor = relay.create_executor("vm", mod=mod, device=dev, target=tgt)
        # (Pdb) type(executor) -- <class 'tvm.relay.backend.vm.VMExecutor'>

        evaluator = executor.evaluate()
        # evaluator -- <function VMExecutor._make_executor.<locals>._vm_wrapper at 0x7f1b87aed620>
        # (Pdb) type(evaluator) -- <class 'function'>

        # Inference
        for name, inp in zip(input_names, input_data):
            params[name] = inp.numpy()

        vm_res = evaluator(**params) # *: Tuple, **: Dict

        # params.keys(), params['i0'].shape -- (dict_keys(['i0']), (10, 20))
        # (Pdb) type(vm_res) -- <class 'tvm.runtime.ndarray.NDArray'>,  vm_res.shape -- (10, 20)

        # Baseline result
        # print("Input shape ...", input_data[0].shape)
        with torch.no_grad():
            pt_result = script_model(*input_data)

        # Verify the accuracy
        # isinstance(pt_result, tuple) -- False
        print("pt_result:", pt_result)
        print("tvm_result:", vm_res)

        if isinstance(pt_result, tuple):
            # handle multiple outputs
            for i in range(len(pt_result)):
                tvm_res = vm_res[i].numpy()
                tvm.testing.assert_allclose(tvm_res, pt_result[i].numpy(), rtol=1e-5, atol=1e-5)
        elif not isinstance(pt_result, torch.Tensor): # isinstance(pt_result, torch.Tensor) -- True
            tvm_res = vm_res.numpy().item()
            assert pt_result == tvm_res
        else:
            tvm.testing.assert_allclose(vm_res.numpy(), pt_result.numpy(), rtol=1e-5, atol=1e-5)
        
        print("Running OK")


class TestMeshgrid(nn.Module):
    def __init__(self):
        super(TestMeshgrid, self).__init__()

    def forward(self, x, y):
        gx, gy = torch.meshgrid([x, y])
        return gx, gy

def test_meshgrid():
    model = TestMeshgrid()
    x = torch.tensor([1, 2, 3, 4]).float()
    y = torch.tensor([4, 5, 6, 7]).float()
    print(model(x, y))
    input_shapes = [('i0', [4]), ('i1', [4])]
    script_model = torch.jit.script(model)
    print(script_model)

    # mod, params = relay.frontend.from_pytorch(script_model, input_shapes)

    verify_model_with_vm(script_model, input_shapes=[(4,), (4,)])


class TorchFlip(nn.Module):
    def __init__(self):
        super(TorchFlip, self).__init__()
        self.params = {}

    def forward(self, x):
        return x.flip([-1])


def test_flip():
    input_shapes = (2, 3, 4)
    model = TorchFlip()
    x = torch.randn(input_shapes)
    print(model(x))

    script_model = torch.jit.script(model)
    print(script_model)

    verify_model_with_vm(script_model, input_shapes=[input_shapes])

# Reference function test_stack_dynamic in tests/python/frontend/pytorch/test_forward.py 
class TorchList(nn.Module):
    def __init__(self):
        super(TorchList, self).__init__()

    def forward(self, x):
        alist = []
        for i in range(5):
            alist += [x]
        return torch.stack(alist, dim=0)


def test_list():
    model = TorchList()
    x = torch.randn(2, 3)
    print(model(x))

    input_shapes = [(2, 3)]
    # trace_model = torch.jit.trace(model, x)
    script_model = torch.jit.script(model)
    print(script_model.graph)

    # mod, params = relay.frontend.from_pytorch(script_model, input_shapes)

    verify_model_with_vm(script_model, input_shapes=[(2, 3)])


class GridSampler(nn.Module):
    def __init__(self, output_h, output_w):
        super(GridSampler, self).__init__()
        self.output_h = output_h
        self.output_w = output_w

        # normalize to [-1, 1]
        h = torch.arange(0, output_h) / (output_h-1) * 2 - 1
        w = torch.arange(0, output_w) / (output_w-1) * 2 - 1
        grid = torch.zeros(output_h, output_w, 2)
        grid[:, :, 0] = w.unsqueeze(0).repeat(output_h, 1)
        grid[:, :, 1] = h.unsqueeze(0).repeat(output_w, 1).transpose(0, 1)
        self.grid = grid.unsqueeze(0)

    def forward(self, input):
        batch = int(input.size(0))
        grid = self.grid.repeat(batch, 1, 1, 1)

        # Torch grid_sample default: mode='bilinear', padding_mode='zeros', align_corners=False
        # tvm seems align corners as True
        return F.grid_sample(input, grid, mode='bilinear', padding_mode='zeros', align_corners=True)


def test_grid_sampler():
    torch.set_grad_enabled(False)

    model = GridSampler(16, 32)
    x = torch.randn(2, 3, 32, 32)
    print(model(x))

    input_shapes = [(2, 3, 32, 32)]
    script_model = torch.jit.script(model)
    print(script_model.graph)

    verify_model_with_vm(script_model, input_shapes=[(2, 3, 32, 32)])



class UnfoldModel(nn.Module):
    def __init__(self):
        super(UnfoldModel, self).__init__()

    def forward(self, input):
        return F.unfold(input, 3, dilation=1, padding=1, stride=1)


def test_unfold():
    torch.set_grad_enabled(False)

    model = UnfoldModel()
    x = torch.randn(2, 3, 16, 16)
    print(model(x))

    input_shapes = [(2, 3, 16, 16)]
    script_model = torch.jit.script(model)
    print(script_model.graph)

    verify_model_with_vm(script_model, input_shapes=[(2, 3, 16, 16)])


def convert_i(i: int) -> float:
    return float(i)

class FloatModel(nn.Module):
    def __init__(self):
        super(FloatModel, self).__init__()

    def forward(self, x):
        f = convert_i(10)
        return f * x

def test_convert():
    torch.set_grad_enabled(False)

    model = FloatModel()
    x = torch.randn(2, 3, 16, 16)
    print(model(x))

    input_shapes = [(2, 3, 16, 16)]
    script_model = torch.jit.script(model)
    print(script_model.graph)

    verify_model_with_vm(script_model, input_shapes=[(2, 3, 16, 16)])


class WhileModel(nn.Module):
    def __init__(self):
        super(WhileModel, self).__init__()

    def forward(self, x):
        # B, C, H, W = x.size(0), x.size(1), x.size(2), x.size(3)
        B, C, H, W = x.size()
        # for r in [-1.0, 1.0]:
        #     for c in [-1.0, 1.0]:
        #         x += r * 10.0
        #         c += c * 1000.0
        for r in range(2):
            for c in range(2):
                r1 = r * 2.0 - 1.0
                c1 = c * 2.0 - 1.0

                x += r1 + c1

        return x

def test_while():
    torch.set_grad_enabled(False)

    model = WhileModel()
    x = torch.randn(2, 3, 16, 16)
    print(model(x))

    input_shapes = [(2, 3, 16, 16)]
    script_model = torch.jit.script(model)
    print(script_model.graph)

    print(script_model.code)

    verify_model_with_vm(script_model, input_shapes=[(2, 3, 16, 16)])


# import tvm
# from tvm import relay
# relay.cast(100, 'float32')



# test_meshgrid()
# test_list()
# test_flip()
# test_grid_sampler()

# test_unfold()

# test_convert()

test_while()


# How to support prim::prim::Uninitialized for frontend ?

# **1. Code patches from pytorch/nn/function.py**
# ```
# def unfold(input, kernel_size, dilation=1, padding=0, stride=1):
#     ...
#     if input.dim() == 4:
#         msg = '{} must be int or 2-tuple for 4D input'
#         assert_int_or_pair(kernel_size, 'kernel_size', msg)
#         assert_int_or_pair(dilation, 'dilation', msg)
#         assert_int_or_pair(padding, 'padding', msg)
#         assert_int_or_pair(stride, 'stride', msg)
#         return torch._C._nn.im2col(input, _pair(kernel_size), _pair(dilation), _pair(padding), _pair(stride))
#     else:
#         raise NotImplementedError("Input Error: Only 4D input Tensors are supported (got {}D)".format(input.dim()))
# ```
# 2. Root cause
#     Convert model with relay.frontend.from_pytorch will generate "prim::Uninitialized", obviously, the above code will break expr.If condition, but " raise NotImplementedError" is population in python language. 
