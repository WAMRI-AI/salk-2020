sample = False
gpu_id = 1
critic_name = '5.3_fastai_80epoch.pkl'
train_date = '5.15'
critic_pretrain = True
expt_name = "mse_transfer"
save_name = '_'.join([train_date, expt_name])

import sys
sys.path.append('../../../model')
sys.path.append('../../../data')
from fastai import *
from fastai.vision import *
from fastai.callbacks import *
from load import get_data, get_patched_data, subsample
from utils import custom_cutout
from metrics import psnr, ssim
import losses

# Basic Setup
torch.cuda.set_device(gpu_id)
data_pth = Path('/home/alaa/Dropbox/BPHO Staff/USF/')
lr_path = f'EM/training/trainsets/lr/'
hr_path = f'EM/training/trainsets/hr/'
critic_pth = data_pth/f'EM/models/self_sv/inpainting'
model_pth = data_pth/f'EM/models/self_sv/baseline'

# Model
def data_func(bs, size, max_zoom):
    func = partial(get_data, data_pth=data_pth, lr_dir=lr_path, hr_dir=hr_path)
    data = func(bs=bs, in_sz=size, out_sz=size, max_zoom=max_zoom)
    if sample:
        return subsample(data)
    else:
        return data

bs_1 = 64
size_1 = 128
db = data_func(bs=bs_1, size=size_1, max_zoom=3)

arch = models.resnet34
wd = 1e-3
superres_metrics = [F.mse_loss, psnr, ssim]
learn = unet_learner(db, arch, wd=wd,
                         loss_func=F.mse_loss,
                         metrics=superres_metrics,
                         blur=True,
                         norm_type=NormType.Weight,
                         model_dir=model_pth)
gc.collect()

## Load Pretrained Inpainting Model
if critic_pretrain:
    learn.model.load_state_dict(torch.load(critic_pth/critic_name))
    print(f"transfer learning from critic: {critic_name}")

# Training - progressive resizing

lr = 1e-3
learn.freeze()
learn.fit_one_cycle(1, lr, pct_start=.9)
learn.unfreeze()
learn.fit_one_cycle(1, slice(1e-5, lr), pct_start=.9)
torch.save(learn.model.state_dict(), model_pth/(save_name+'_1.pkl'))


bs_2 = 16
size_2 = 256
db = data_func(bs=bs_2, size=size_2, max_zoom=3.)
learn = unet_learner(db, arch, wd=wd,
                         loss_func=feat_loss,
                         metrics=superres_metrics,
                         blur=True,
                         norm_type=NormType.Weight,
                         model_dir=model_pth)
learn.model.load_state_dict(torch.load(model_pth/(save_name + '_1.pkl')))
learn.freeze()
learn.fit_one_cycle(3, lr, pct_start=.9)
learn.unfreeze()
learn.fit_one_cycle(3, slice(1e-5, lr), pct_start=.9)
torch.save(learn.model.state_dict(), model_pth/(save_name + '_2.pkl'))

bs_3 = 8
size_3 = 512
db = data_func(bs=bs_3, size=size_3, max_zoom=2.)
learn = unet_learner(db, arch, wd=wd,
                         loss_func=feat_loss,
                         metrics=superres_metrics,
                         blur=True,
                         norm_type=NormType.Weight,
                         model_dir=model_pth)
learn.model.load_state_dict(torch.load(model_pth/save_name + '_2.pkl'))
learn.freeze()
learn.fit_one_cycle(3, lr, pct_start=.9)
learn.unfreeze()
learn.fit_one_cycle(3, slice(1e-5, lr/10), pct_start=.9)
torch.save(learn.model.state_dict(), model_pth/(save_name + '_3.pkl'))