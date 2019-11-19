import sys
import math
import torch
import torch.nn as nn
from layers import conv_layer


class ResBlock(nn.Module):
    def __init__(self, expansion, cin, chid, stride=1):
        super().__init__()
        cout, cin = chid*expansion, cin*expansion
        layers = [conv_layer(cin, chid, 1)]
        if expansion==1:
            layers += [conv_layer(cin, cout, 3, stride=stride, act=False)]
        else:
            layers += [conv_layer(cin, chid, 3, stride=stride, act=True),
                       conv_layer()]
