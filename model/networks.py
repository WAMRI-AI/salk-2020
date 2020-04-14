import sys
import math
from .blocks import SimpleUnetBlock
from fastai import *
from fastai.vision import *
from fastai.callbacks import *
from torchvision.models.resnet import BasicBlock, conv1x1

def custom_resnet34(c_in=3):
    """Customizes the number of input channels of resnet34"""
    return CustomResNet(c_in, BasicBlock, [3, 4, 6, 3])

class CustomResNet(nn.Module):
    """Assgin only one channel to the first layer of ResNet"""
    def __init__(self, c_in, block, layers, num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None):
        super(CustomResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(c_in, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def _forward_impl(self, x):
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

    def forward(self, x):
        return self._forward_impl(x)

class SimpleDynamicUnet(SequentialEx):
    "Create a U-Net from a given architecture."
    def __init__(self, encoder:nn.Module, n_classes:int, 
                 img_size:Tuple[int,int]=(256,256), blur:bool=False, 
                 blur_final=True, self_attention:bool=False,
                 y_range:Optional[Tuple[float,float]]=None,
                 last_cross:bool=True, bottle:bool=False, **kwargs):
        imsize = img_size
        # get output sizes based on input size
        sfs_szs = model_sizes(encoder, size=imsize)
        # get layer idxs where output size change
        sfs_idxs = list(reversed(_get_sfs_idxs(sfs_szs)))
        # grab activation maps from those layers
        self.sfs = hook_outputs([encoder[i] for i in sfs_idxs], detach=False)
        # get dummy output from encoder (to init linker layers)
        x = dummy_eval(encoder, imsize).detach()
        ni = sfs_szs[-1][1]
        # Linker layers between encoder and decoder
        middle_conv = nn.Sequential(conv_layer(ni, ni*2, **kwargs),
                                    conv_layer(ni*2, ni, **kwargs)).eval()
        x = middle_conv(x)
        # collate all the layers
        layers = [encoder, batchnorm_2d(ni), nn.ReLU(), middle_conv]
        
        for i,idx in enumerate(sfs_idxs):
            # is this final u-net block? True/False
            not_final = i!=len(sfs_idxs)-1
            # num channels for upsampling input and act map from encoder
            up_in_c, x_in_c = int(x.shape[1]), int(sfs_szs[idx][1])
            # blurring if use blur and not final u-net block
            do_blur = blur and (not_final or blur_final)
            # attn if use attn and 1st act map (?)
            sa = self_attention and (i==len(sfs_idxs)-3)
            # apply a U-net block for Decoder
            unet_block = SimpleUnetBlock(up_in_c, 
                                         self.sfs[i], 
                                         final_div=not_final, 
                                         blur=do_blur, self_attention=sa,
                                         **kwargs).eval()
            # add to your layers
            layers.append(unet_block)
            # get output (for shape reasons later on...)
            x = unet_block(x)
        # n_channels of our current feature map
        ni = x.shape[1]
        if imsize != sfs_szs[0][-2:]: layers.append(PixelShuffle_ICNR(ni, **kwargs))
        x = PixelShuffle_ICNR(ni)(x)
        if imsize != x.shape[-2:]: layers.append(Lambda(lambda x: F.interpolate(x, imsize, mode='nearest')))
        if last_cross:
            layers.append(MergeLayer(dense=True))
            # in channels of 1st encoder layer
            ni += in_channels(encoder)
            layers.append(res_block(ni, bottle=bottle, **kwargs))
        # final conv for predictions
        layers += [conv_layer(ni, n_classes, ks=1, use_activ=False, **kwargs)]
        if y_range is not None: layers.append(SigmoidRange(*y_range))
        super().__init__(*layers)

        
        
def _get_sfs_idxs(sizes:Sizes) -> List[int]:
    "Get the indexes of the layers where the size of the activation changes."
    feature_szs = [size[-1] for size in sizes]
    sfs_idxs = list(np.where(np.array(feature_szs[:-1])!=np.array(feature_szs[1:]))[0])
    if feature_szs[0] != feature_szs[1]: sfs_idxs = [0] + sfs_idxs
    return sfs_idxs

# class DynamicUnet(SequentialEx):
#     "Create a U-Net from a given architecture."
#     def __init__(self, encoder, n_classes, img_size,
#                  blur=False, blur_final=True, self_attention=False,
#                  y_range=None, last_cross=True, bottle=False,
#                  act_cls=defaults.activation, init=nn.init.kaiming_normal_, 
#                  norm_type=NormType.Batch, **kwargs):
#         imsize = img_size
#         # get output sizes based on input size
#         sizes = model_sizes(encoder, size=imsize)
#         # get layer idxs where output size change
#         sz_chg_idxs = list(reversed(_get_sz_change_idxs(sizes)))
#         # grab activation maps from those layers
#         self.sfs = hook_outputs([encoder[i] for i in sz_chg_idxs], detach=False)
#         # get dummy output from encoder (to init linker layers)
#         x = dummy_eval(encoder, imsize).detach()
        
#         ni = sizes[-1][1]
#         # Linker layers between encoder and decoder
#         middle_conv = nn.Sequential(ConvLayer(ni, ni*2, act_cls=act_cls,
#                                               norm_type=norm_type, **kwargs),
#                                     ConvLayer(ni*2, ni, act_cls=act_cls,
#                                               norm_type=norm_type, **kwargs)).eval()
#         # init Linker layer
#         x = middle_conv(x)
#         layers = [encoder, BatchNorm(ni), nn.ReLU(), middle_conv]

#         for i,idx in enumerate(sz_chg_idxs):
#             # is this the final u-net block?
#             not_final = i!=len(sz_chg_idxs)-1
#             # num channels for upsampling input and act map from encoder
#             up_in_c, x_in_c = int(x.shape[1]), int(sizes[idx][1])
#             # blurring if use blur and not final u-net block
#             do_blur = blur and (not_final or blur_final)
#             # attn if use attn and 1st act map (?)
#             sa = self_attention and (i==len(sz_chg_idxs)-3)
#             # apply a U-net block for Decoder
#             unet_block = UnetBlock(up_in_c, x_in_c, self.sfs[i], 
#                                    final_div=not_final, blur=do_blur,
#                                    self_attention=sa, act_cls=act_cls, 
#                                    init=init, norm_type=norm_type, **kwargs).eval()
#             layers.append(unet_block)
#             x = unet_block(x)
        
#         ni = x.shape[1]
        
#         if imsize != sizes[0][-2:]:
#             pxl_shuff = PixelShuffle_ICNR(ni, act_cls=act_cls, norm_type=norm_type)
#             layers.append(pxl_shuff)
#         x = PixelShuffle_ICNR(ni)(x)
#         if imsize != x.shape[-2:]: 
#             layers.append(Lambda(lambda x: F.interpolate(x, imsize, mode='nearest')))
#         if last_cross:
#             layers.append(MergeLayer(dense=True))
#             ni += in_channels(encoder)
#             res_block = ResBlock(1, ni, ni//2 if bottle else ni, 
#                                  act_cls=act_cls, norm_type=norm_type, 
#                                  **kwargs)
#             layers.append(res_block)
#         conv_layer = ConvLayer(ni, n_classes, ks=1, 
#                                act_cls=None, norm_type=norm_type, 
#                                **kwargs)
#         layers += [conv_layer]
#         apply_init(nn.Sequential(layers[3], layers[-2]), init)
#         #apply_init(nn.Sequential(layers[2]), init)
#         if y_range is not None:
#             layers.append(SigmoidRange(*y_range))
#         super().__init__(*layers)

#     def __del__(self):
#         if hasattr(self, "sfs"): self.sfs.remove()