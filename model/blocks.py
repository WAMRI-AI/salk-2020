import sys
import math
from fastai import *
from fastai.vision import *
from fastai.callbacks import *


class SimpleUnetBlock(Module):
    "A quasi-UNet block, using `PixelShuffle_ICNR upsampling`."
    def __init__(self, up_in_c:int, 
                 hook:Hook, 
                 final_div:bool=True, blur:bool=False, leaky:float=None,
                 self_attention:bool=False, **kwargs):
        self.shuf = PixelShuffle_ICNR(up_in_c, up_in_c//2, blur=blur, 
                                      leaky=leaky, **kwargs)
        # self.bn = batchnorm_2d(x_in_c)
        self.hook = hook
        ni = up_in_c//2 
        nf = ni if final_div else ni//2
        self.conv1 = conv_layer(ni, nf, leaky=leaky, **kwargs)
        self.conv2 = conv_layer(nf, nf, leaky=leaky, self_attention=self_attention, 
                                **kwargs)
        self.relu = relu(leaky=leaky)

    def forward(self, up_in:Tensor) -> Tensor:
        s = self.hook.stored
        up_out = self.shuf(up_in)
        ssh = s.shape[-2:]
        if ssh != up_out.shape[-2:]:
            up_out = F.interpolate(up_out, s.shape[-2:], mode='nearest')
        # cat_x = self.relu(torch.cat([up_out, self.bn(s)], dim=1))
        return self.conv2(self.conv1(self.relu(up_out)))


# class UnetBlock(Module):
#     "A quasi-UNet block, using `PixelShuffle_ICNR upsampling`."
#     def __init__(self, up_in_c:int, x_in_c:int, hook:Hook, final_div:bool=True, blur:bool=False, leaky:float=None,
#                  self_attention:bool=False, **kwargs):
#         self.hook = hook
#         self.shuf = PixelShuffle_ICNR(up_in_c, up_in_c//2, blur=blur, leaky=leaky, **kwargs)
#         self.bn = batchnorm_2d(x_in_c)
#         ni = up_in_c//2 + x_in_c
#         nf = ni if final_div else ni//2
#         self.conv1 = conv_layer(ni, nf, leaky=leaky, **kwargs)
#         self.conv2 = conv_layer(nf, nf, leaky=leaky, self_attention=self_attention, **kwargs)
#         self.relu = relu(leaky=leaky)

#     def forward(self, up_in:Tensor) -> Tensor:
#         s = self.hook.stored
#         up_out = self.shuf(up_in)
#         ssh = s.shape[-2:]
#         if ssh != up_out.shape[-2:]:
#             up_out = F.interpolate(up_out, s.shape[-2:], mode='nearest')
#         cat_x = self.relu(torch.cat([up_out, self.bn(s)], dim=1))
#         return self.conv2(self.conv1(cat_x))
    
    
# class ResBlock(nn.Module):
#     "Resnet block from `ni` to `nh` with `stride`"
#     @delegates(ConvLayer.__init__)
#     def __init__(self, expansion, ni, nf, stride=1, 
#                  groups=1, reduction=None, nh1=None, 
#                  nh2=None, dw=False, g2=1, sa=False, 
#                  sym=False, norm_type=NormType.Batch, 
#                  act_cls=defaults.activation, ndim=2, ks=3,
#                  pool=AvgPool, pool_first=True, **kwargs):
#         super().__init__()
#         norm2 = (NormType.BatchZero if norm_type==NormType.Batch else
#                  NormType.InstanceZero if norm_type==NormType.Instance \ 
#                  else norm_type)
#         if nh2 is None: nh2 = nf
#         if nh1 is None: nh1 = nh2
#         nf,ni = nf*expansion,ni*expansion
#         k0 = dict(norm_type=norm_type, act_cls=act_cls, ndim=ndim, **kwargs)
#         k1 = dict(norm_type=norm2, act_cls=None, ndim=ndim, **kwargs)
#         convpath  = [ConvLayer(ni,  nh2, ks, stride=stride, 
#                                groups=ni if dw else groups, **k0),
#                      ConvLayer(nh2,  nf, ks, groups=g2, **k1)
#         ] if expansion == 1 else [
#                      ConvLayer(ni,  nh1, 1, **k0),
#                      ConvLayer(nh1, nh2, ks, stride=stride, 
#                                groups=nh1 if dw else groups, **k0),
#                      ConvLayer(nh2,  nf, 1, groups=g2, **k1)]
#         if reduction: convpath.append(SEModule(nf, reduction=reduction,
#                                                act_cls=act_cls))
#         if sa: convpath.append(SimpleSelfAttention(nf,ks=1,sym=sym))
#         self.convpath = nn.Sequential(*convpath)
#         idpath = []
#         if ni!=nf: idpath.append(ConvLayer(ni, nf, 1, act_cls=None, 
#                                            ndim=ndim, **kwargs))
#         if stride!=1: idpath.insert((1,0)[pool_first], 
#                                     pool(2, ndim=ndim, ceil_mode=True))
#         self.idpath = nn.Sequential(*idpath)
#         self.act = defaults.activation(inplace=True) if act_cls is defaults.activation else act_cls()

#     def forward(self, x): return self.act(self.convpath(x) + self.idpath(x))