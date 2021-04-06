import os

import torch
from torch import nn
from torch.nn import functional as F

module_path = os.path.dirname(__file__)



class FusedLeakyReLU(nn.Module):
    def __init__(self, channel, negative_slope=0.2, scale=2 ** 0.5):
        super().__init__()

        self.bias = nn.Parameter(torch.zeros(channel))
        self.negative_slope = negative_slope
        self.scale = scale

    def forward(self, input):
        return fused_leaky_relu(input, self.bias, self.negative_slope, self.scale)


def fused_leaky_relu(input, bias, negative_slope=0.2, scale=2 ** 0.5):
    rest_dim = [1] * (input.ndim - bias.ndim - 1)
    input = input.cuda()
    if input.ndim == 3:
        return (
            F.leaky_relu(
                input + bias.view(1, *rest_dim, bias.shape[0]), negative_slope=negative_slope
            )
            * scale
        )
    else:
        return (
            F.leaky_relu(
                input + bias.view(1, bias.shape[0], *rest_dim), negative_slope=negative_slope
            )
            * scale
        )

