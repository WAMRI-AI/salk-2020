from fastai import *
from fastai.vision import *
from fastai.callbacks import *
from .networks import SimpleDynamicUnet


def _resnet_split(m:nn.Module): return (m[0][6],m[1])

def simple_unet_learner(data:DataBunch, arch:Callable, pretrained:bool=True, 
                        blur_final:bool=True, norm_type:Optional[NormType]=None, 
                        split_on:Optional[SplitFuncOrIdxList]=None, blur:bool=False,
                        self_attention:bool=False, 
                        y_range:Optional[Tuple[float,float]]=None, 
                        last_cross:bool=False,
                        bottle:bool=False, cut:Union[int,Callable]=None, 
                        **learn_kwargs:Any)->Learner:
    "Build Unet learner from `data` and `arch`."
    meta = {'cut': -2, 'split': _resnet_split}
    body = create_body(arch, pretrained, cut)
    try:    size = data.train_ds[0][0].size
    except: size = next(iter(data.train_dl))[0].shape[-2:]
    model = to_device(SimpleDynamicUnet(body, n_classes=data.c, img_size=size, 
                                        blur=blur, blur_final=blur_final,
                                        self_attention=self_attention, 
                                        y_range=y_range, norm_type=norm_type,
                                        last_cross=last_cross,
                                        bottle=bottle), data.device)
    learn = Learner(data, model, **learn_kwargs)
    learn.split(ifnone(split_on, meta['split']))
    if pretrained: learn.freeze()
    apply_init(model[2], nn.init.kaiming_normal_)
    return learn