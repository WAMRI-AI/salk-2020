from fastai import *
from fastai.vision import *
from fastai.callbacks import *


import sys
sys.path.append('../../')
from data.load import get_data
from model.metrics import ssim, psnr

torch.cuda.set_device(1)

nb_name = 'al-pretrained_hybrid'
data_pth = Path('/home/alaa/Dropbox/BPHO Staff/USF/')
lr_path = f'EM/training/trainsets/lr/'
svd_path = f'EM/training/trainsets/svd_30_40/'
hr_path = f'EM/training/trainsets/hr/'
model_path = data_pth/f'EM/models/crappifiers/'

# model setup
arch = models.resnet34
wd = 1e-3
superres_metrics = [F.l1_loss, F.mse_loss, ssim, psnr]

# helper func
lr = 1e-3
def do_fit(save_name, lrs=slice(lr), pct_start=0.9, cycle_len=10):
    learn.fit_one_cycle(cycle_len, lrs, pct_start=pct_start)
    learn.save(save_name)

# loading 3 rounds of data

# Round 1 - Original Crappifier
bs_1 = 64
size_1 = 128
db_1_lr = get_data(data_pth=data_pth, lr_dir=lr_path, hr_dir=hr_path,
             bs=bs_1, in_sz=size_1, out_sz=size_1, max_zoom=6)

# Round 1 - SVD Crappifier
db_1_svd = get_data(data_pth=data_pth, lr_dir=svd_path, hr_dir=hr_path,
             bs=bs_1, in_sz=size_1, out_sz=size_1, max_zoom=6)


# Round 2 - Original Crappifier
bs_2 = 16
size_2 = 256
db_2_lr = get_data(data_pth=data_pth, lr_dir=lr_path, hr_dir=hr_path,
             bs=bs_2, in_sz=size_2, out_sz=size_2, max_zoom=3)

# Round 2 - SVD Crappifier
db_2_svd = get_data(data_pth=data_pth, lr_dir=svd_path, hr_dir=hr_path,
             bs=bs_2, in_sz=size_2, out_sz=size_2, max_zoom=3)


# Round 3 - Original Crappifier
bs_3 = 8
size_3 = 512
db_3_lr = get_data(data_pth=data_pth, lr_dir=lr_path, hr_dir=hr_path,
             bs=bs_3, in_sz=size_3, out_sz=size_3, max_zoom=2.)

# Round 3 - SVD Crappifier
db_3_svd = get_data(data_pth=data_pth, lr_dir=svd_path, hr_dir=hr_path,
             bs=bs_3, in_sz=size_3, out_sz=size_3, max_zoom=2.)


# fitting round 1
learn = unet_learner(db_1_lr, arch, 
                     wd=wd, 
                     #loss_func=feat_loss,
                     loss_func=F.mse_loss,
                     metrics=superres_metrics, 
                     #callback_fns=LossMetrics, 
                     blur=True, norm_type=NormType.Weight, model_dir=model_path)
do_fit(f'{nb_name}.1a', 1e-3, cycle_len=1)
learn.data = db_1_svd
do_fit(f'{nb_name}.1b', 1e-3, cycle_len=1)

learn.unfreeze()
do_fit(f'{nb_name}.1c', slice(1e-5,1e-3), cycle_len=1)
learn.data = db_1_lr
do_fit(f'{nb_name}.1d', slice(1e-5,1e-3), cycle_len=1)
del(learn)
gc.collect()

# fitting round 2
learn = unet_learner(db_2_lr, arch, 
                     wd=wd, 
                     #loss_func=feat_loss,
                     loss_func=F.mse_loss,
                     metrics=superres_metrics, 
                     #callback_fns=LossMetrics, 
                     blur=True, norm_type=NormType.Weight, model_dir=model_path)
learn.load(f'{nb_name}.1d')
do_fit(f'{nb_name}.2a', 1e-3, cycle_len=3)
learn.data = db_2_svd
do_fit(f'{nb_name}.2b', 1e-3, cycle_len=3)
learn.unfreeze()
do_fit(f'{nb_name}.2c', slice(1e-5, 1e-3), cycle_len=3)
learn.data = db_2_lr
do_fit(f'{nb_name}.2d', slice(1e-5, 1e-3), cycle_len=3)
del(learn)
gc.collect()

# fitting round 3
learn = unet_learner(db_3_lr, arch, 
                     wd=wd, 
                     #loss_func=feat_loss,
                     loss_func=F.mse_loss,
                     metrics=superres_metrics, 
                     #callback_fns=LossMetrics, 
                     blur=True, norm_type=NormType.Weight, model_dir=model_path)
learn.load(f'{nb_name}.2d')
do_fit(f'{nb_name}.3a', lr, cycle_len=3)
learn.data = db_3_svd
do_fit(f'{nb_name}.3b', lr, cycle_len=3)
learn.unfreeze()
do_fit(f'{nb_name}.3c', slice(1e-5, 1e-4), cycle_len=3)
learn.data = db_3_lr
do_fit(f'{nb_name}.3d', slice(1e-5, 1e-4), cycle_len=3)
print("PSSR using Hybrid Crappification Experiment Complete")