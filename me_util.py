# Copyright (c) Chris Choy (chrischoy@ai.stanford.edu). All Rights Reserved.
#
# Please cite "4D Spatio-Temporal ConvNets: Minkowski Convolutional Neural
# Networks", CVPR'19 (https://arxiv.org/abs/1904.08755) if you use any part of
# the code.
from typing import Union
from enum import Enum
import torch.nn as nn
import MinkowskiEngine as ME
import MinkowskiEngine.MinkowskiFunctional as MEF
def get_norm(norm_type: str, n_channels: int, D: int, bn_momentum: float = 0.1):
    if norm_type == "BN":
        return ME.MinkowskiBatchNorm(n_channels, momentum=bn_momentum)
    elif norm_type == "IN":
        return ME.MinkowskiInstanceNorm(n_channels, dimension=D)
    else:
        raise ValueError(f"Norm type: {norm_type} not supported")
NONLINEARITIES = [
    ME.MinkowskiReLU,
    ME.MinkowskiPReLU,
    ME.MinkowskiLeakyReLU,
    ME.MinkowskiELU,
    ME.MinkowskiCELU,
    ME.MinkowskiSELU,
    ME.MinkowskiGELU,
]
NONLINEARITIES_dict = {i: n for i, n in enumerate(NONLINEARITIES)}
for n in NONLINEARITIES:
    NONLINEARITIES_dict[n.__name__] = n
def get_nonlinearity(nonlinearity_type: Union[str, int]):
    return NONLINEARITIES_dict[nonlinearity_type]
def get_nonlinearity_fn(
    nonlinearity_type: str, input: ME.SparseTensor, *args, **kwargs
):
    if nonlinearity_type == "MinkowskiReLU":
        return MEF.relu(input, *args, **kwargs)
    elif nonlinearity_type == "MinkowskiLeakyReLU":
        return MEF.leaky_relu(input, *args, **kwargs)
    elif nonlinearity_type == "MinkowskiPReLU":
        return MEF.prelu(input, *args, **kwargs)
    elif nonlinearity_type == "MinkowskiCELU":
        return MEF.celu(input, *args, **kwargs)
    elif nonlinearity_type == "MinkowskiSELU":
        return MEF.selu(input, *args, **kwargs)
    elif nonlinearity_type == "MinkowskiGELU":
        return MEF.gelu(input, *args, **kwargs)
    else:
        raise ValueError(f"Norm type: {nonlinearity_type} not supported")
def conv(in_planes, out_planes, kernel_size, stride=1, dilation=1, bias=False, D=-1):
    assert D > 0, "Dimension must be a positive integer"
    return ME.MinkowskiConvolution(
        in_channels=in_planes,
        out_channels=out_planes,
        kernel_size=kernel_size,
        stride=stride,
        dilation=dilation,
        bias=bias,
        dimension=D,
    )
def conv_tr(in_planes, out_planes, kernel_size, stride=1, dilation=1, bias=False, D=-1):
    assert D > 0, "Dimension must be a positive integer"
    return ME.MinkowskiConvolutionTranspose(
        in_channels=in_planes,
        out_channels=out_planes,
        kernel_size=kernel_size,
        stride=stride,
        dilation=dilation,
        bias=bias,
        dimension=D,
    )
def block(
    in_channel,
    out_channel,
    kernel_size,
    stride,
    bn_momentum=0.05,
    norm_type="BN",
    nonlinearity_type="MinkowskiReLU",
    D=-1,
):
    return nn.Sequential(
        conv(in_channel, out_channel, kernel_size=kernel_size, stride=stride, D=D),
        get_norm(norm_type, out_channel, D=D, bn_momentum=bn_momentum),
        get_nonlinearity(nonlinearity_type)(),
    )
def block_tr(
    in_channel,
    out_channel,
    kernel_size,
    stride,
    bn_momentum=0.05,
    norm_type="BN",
    nonlinearity_type="MinkowskiReLU",
    D=-1,
):
    return nn.Sequential(
        conv_tr(in_channel, out_channel, kernel_size=kernel_size, stride=stride, D=D),
        get_norm(norm_type, out_channel, D=D, bn_momentum=bn_momentum),
        get_nonlinearity(nonlinearity_type)(),
    )