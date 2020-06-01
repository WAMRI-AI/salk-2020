gpu_id = 2
train_date = '5.21'

critic_name = '5.3_fastai_80epoch.pkl'
critic_transfer = False
expt_name = "inpaint_imagenet_toddler"

sample = True
expt_name += '_sampled'

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
model_pth = data_pth/f'EM/models/self_sv/inpainting'

# Resnet Feature loss
bs = 8  # batch size
size = 256  # image size
random_patch = partial(custom_cutout, min_n_holes=10, max_n_holes=20,
                       min_length=15, max_length=25, use_on_y=False)
tfms = [[random_patch()], [random_patch()]]
data = get_patched_data(data_pth, hr_path, bs, tfms=tfms,
                        in_sz=size, out_sz=size)
critic_arch = models.resnet34
wd = 1e-3
superres_metrics = [F.mse_loss, psnr, ssim]
critic = unet_learner(data, critic_arch, wd=wd,
                         loss_func=F.mse_loss,
                         metrics=superres_metrics,
                         blur=True,
                         norm_type=NormType.Weight)
gc.collect()
critic.model.load_state_dict(torch.load(model_pth/critic_name))
encoder = critic.model.eval()[0]
feature_maps = losses.find_layers(flatten_model(encoder))
num_layers = len(feature_maps)
feat_loss = losses.FeatureLoss(m_feat=encoder, layer_wgts=[1/num_layers for _ in range(num_layers)])

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
learn = unet_learner(db, arch, wd=wd,
                         loss_func=feat_loss,
                         metrics=superres_metrics,
                         blur=True,
                         norm_type=NormType.Weight,
                         model_dir=model_pth)
gc.collect()

## Load Pretrained Inpainting Model
if critic_transfer:
    learn.model.load_state_dict(torch.load(model_pth/critic_name))
    print("transfer learning from critic")

# Training - progressive resizing

lr = 1e-3
learn.freeze()
learn.fit_one_cycle(1, lr, pct_start=.9)
learn.unfreeze()
learn.fit_one_cycle(1, slice(1e-5, lr), pct_start=.9)
torch.save(learn.model.state_dict(), model_pth/(save_name + '_1.pkl'))


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
learn.model.load_state_dict(torch.load(model_pth/(save_name + '_2.pkl')))
learn.freeze()
learn.fit_one_cycle(3, lr, pct_start=.9)
learn.unfreeze()
learn.fit_one_cycle(3, slice(1e-5, lr/10), pct_start=.9)
torch.save(learn.model.state_dict(), model_pth/(save_name + '_3.pkl'))