import sys
import math
import torch
import torch.nn as nn
from activations import relu, gelu, sigmoid


def conv(cin, cout, k=3, stride=1, bias=False):
    return nn.Conv2d(cin, cout, kernel_size=k, stride=stride,
                     padding=k//2, bias=bias)


def wn_conv(cin, cout, k=3, stride=1, act=True):
    layers = [nn.utils.weight_norm(conv(cin, cout, k, stride=stride))]
    if act: layers.append(relu)
    return nn.Sequential(*layers)


def noop(x): return x


class Flatten(nn.Module):
    def forward(self, x): return x.view(x.size(0), -1)


# sequential takes list of params
# *layers deconstructs list to individual items
