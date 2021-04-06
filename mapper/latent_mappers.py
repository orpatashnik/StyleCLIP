import torch
from torch import nn
from torch.nn import Module

from models.stylegan2.model import EqualLinear, PixelNorm


class Mapper(Module):

    def __init__(self, opts):
        super(Mapper, self).__init__()

        self.opts = opts
        layers = [PixelNorm()]

        for i in range(4):
            layers.append(
                EqualLinear(
                    512, 512, lr_mul=0.01, activation='fused_lrelu'
                )
            )

        self.mapping = nn.Sequential(*layers)


    def forward(self, x):
        x = self.mapping(x)
        return x


class SingleMapper(Module):

    def __init__(self, opts):
        super(SingleMapper, self).__init__()

        self.opts = opts

        self.mapping = Mapper(opts)

    def forward(self, x):
        out = self.mapping(x)
        return out


class LevelsMapper(Module):

    def __init__(self, opts):
        super(LevelsMapper, self).__init__()

        self.opts = opts

        if not opts.no_coarse_mapper:
            self.course_mapping = Mapper(opts)
        if not opts.no_medium_mapper:
            self.medium_mapping = Mapper(opts)
        if not opts.no_fine_mapper:
            self.fine_mapping = Mapper(opts)

    def forward(self, x):
        x_coarse = x[:, :4, :]
        x_medium = x[:, 4:8, :]
        x_fine = x[:, 8:, :]

        if not self.opts.no_coarse_mapper:
            x_coarse = self.course_mapping(x_coarse)
        else:
            x_coarse = torch.zeros_like(x_coarse)
        if not self.opts.no_medium_mapper:
            x_medium = self.medium_mapping(x_medium)
        else:
            x_medium = torch.zeros_like(x_medium)
        if not self.opts.no_fine_mapper:
            x_fine = self.fine_mapping(x_fine)
        else:
            x_fine = torch.zeros_like(x_fine)


        out = torch.cat([x_coarse, x_medium, x_fine], dim=1)

        return out

