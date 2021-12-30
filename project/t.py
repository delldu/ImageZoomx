"""Onnx Model Tools."""
# coding=utf-8

import torch
import torch.nn as nn

import pdb


class MicroModel(nn.Module):
    def __init__(self):
        """Init model."""
        super(MicroModel, self).__init__()
        self.conv = nn.Conv2d(3, 8, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        return x


def onnx_export():
    model = MicroModel()
    model = model.eval()

    x = torch.randn(1, 3, 256, 256)
    with torch.no_grad():
        y = model(x)

    torch.onnx.export(
        model,
        x,
        "micro.onnx",
        input_names=["input"],
        output_names=["output"],
        verbose=True,
        opset_version=11,
        keep_initializers_as_inputs=False,
        export_params=True,
    )


onnx_export()
