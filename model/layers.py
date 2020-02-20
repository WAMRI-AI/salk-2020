import sys
import math
import torch
import torch.nn as nn
from activations import relu, gelu, sigmoid


class ConvLayer(nn.Sequential):
    "Create a sequence of convolutional (`ni` to `nf`), ReLU (if `use_activ`) and `norm_type` layers."
    def __init__(self, ni, nf, ks=3, stride=1, padding=None,
                 bias=None, ndim=2, norm_type=NormType.Batch, bn_1st=True,
                 act_cls=defaults.activation, transpose=False, 
                 init=nn.init.kaiming_normal_, xtra=None, **kwargs):
        if padding is None: padding = ((ks-1)//2 if not transpose else 0)
        bn = norm_type in (NormType.Batch, NormType.BatchZero)
        if bias is None: bias = not bn
        conv_func = _conv_func(ndim, transpose=transpose)
        conv = init_default(conv_func(ni, nf, kernel_size=ks, bias=bias, 
                                      stride=stride, padding=padding, **kwargs), 
                            init)
        if   norm_type==NormType.Weight:   conv = weight_norm(conv)
        elif norm_type==NormType.Spectral: conv = spectral_norm(conv)
        layers = [conv]
        act_bn = []
        if act_cls is not None: act_bn.append(act_cls())
        if bn: act_bn.append(BatchNorm(nf, norm_type=norm_type, ndim=ndim))
        if bn_1st: act_bn.reverse()
        layers += act_bn
        if xtra: layers.append(xtra)
        super().__init__(*layers)


class MergeLayer(nn.Module):
    "Merge a shortcut with the result of the module by adding them or \ 
    concatenating them if `dense=True`."
    def __init__(self, dense:bool=False): self.dense=dense
    def forward(self, x): 
        return torch.cat([x,x.orig], dim=1) if self.dense else (x+x.orig)


def batchnorm_2d(nf:int, norm_type:NormType=NormType.Batch):
    "A batchnorm2d layer with `nf` features initialized depending on `norm_type`."
    bn = nn.BatchNorm2d(nf)
    with torch.no_grad():
        bn.bias.fill_(1e-3)
        bn.weight.fill_(0. if norm_type==NormType.BatchZero else 1.)
    return bn


class PixelShuffle_ICNR(nn.Sequential):
    "Upsample by `scale` from `ni` filters to `nf` (default `ni`), using `nn.PixelShuffle`."
    def __init__(self, ni, nf=None, scale=2, blur=False,
                 norm_type=NormType.Weight,
                 act_cls=defaults.activation):
        super().__init__()
        nf = ifnone(nf, ni)
        layers = [ConvLayer(ni, nf*(scale**2), ks=1, 
                            norm_type=norm_type, act_cls=act_cls),
                  nn.PixelShuffle(scale)]
        layers[0][0].weight.data.copy_(icnr_init(layers[0][0].weight.data))
        if blur: layers += [nn.ReplicationPad2d((1,0,1,0)), 
                            nn.AvgPool2d(2, stride=1)]
        super().__init__(*layers)

        
class SelfAttention(nn.Module):
    "Self attention layer for `n_channels`."
    def __init__(self, n_channels):
        super().__init__()
        self.query,self.key,self.value = [self._conv(n_channels, c) for c in \ 
                                          (n_channels//8,n_channels//8,n_channels)]
        
        self.gamma = nn.Parameter(tensor([0.]))

    def _conv(self,n_in,n_out):
        return ConvLayer(n_in, n_out, ks=1, ndim=1,
                         norm_type=NormType.Spectral,
                         act_cls=None, bias=False)

    def forward(self, x):
        #Notation from the paper.
        size = x.size()
        x = x.view(*size[:2],-1)
        f,g,h = self.query(x),self.key(x),self.value(x)
        beta = F.softmax(torch.bmm(f.transpose(1,2), g), dim=1)
        o = self.gamma * torch.bmm(h, beta) + x
        return o.view(*size).contiguous()      


class SimpleSelfAttention(Module):
    def __init__(self, n_in:int, ks=1, sym=False):
        self.sym,self.n_in = sym,n_in
        self.conv = _conv1d_spect(n_in, n_in, ks, padding=ks//2, bias=False)
        self.gamma = nn.Parameter(tensor([0.]))

    def forward(self,x):
        if self.sym:
            c = self.conv.weight.view(self.n_in,self.n_in)
            c = (c + c.t())/2
            self.conv.weight = c.view(self.n_in,self.n_in,1)

        size = x.size()
        x = x.view(*size[:2],-1)

        convx = self.conv(x)
        xxT = torch.bmm(x,x.permute(0,2,1).contiguous())
        o = torch.bmm(xxT, convx)
        o = self.gamma * o + x
        return o.view(*size).contiguous()
    
    
def SEModule(ch, reduction, act_cls=defaults.activation):
    nf = math.ceil(ch//reduction/8)*8
    return SequentialEx(nn.AdaptiveAvgPool2d(1),
                        ConvLayer(ch, nf, ks=1, norm_type=None, act_cls=act_cls),
                        ConvLayer(nf, ch, ks=1, norm_type=None, act_cls=nn.Sigmoid),
                        ProdLayer())

    
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
