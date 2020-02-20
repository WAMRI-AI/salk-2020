import sys
import math
from .blocks import SimpleUnetBlock
from fastai import *
from fastai.vision import *
from fastai.callbacks import *



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