from fastai import *
from fastai.vision import *
from fastai.callbacks import *


import sys
sys.path.append('/home/rusl/Documents/salk-practicum')
from data.load import get_data
from model.metrics import ssim, psnr

torch.cuda.set_device(2)

nb_name = 'annette-pretrained'
data_pth = Path('/home/alaa/Dropbox (BPHO)/BPHO Staff/USF/')
lr_path = f'EM/training/trainsets/crappified/'
hr_path = f'EM/training/trainsets/hr/'
model_path = data_pth/f'EM/models/'


# model setup
arch = models.resnet34
wd = 1e-3
superres_metrics = [F.mse_loss, ssim, psnr]

# helper func
lr = 1e-3
def do_fit(save_name, lrs=slice(lr), pct_start=0.9, cycle_len=10):
    learn.fit_one_cycle(cycle_len, lrs, pct_start=pct_start)
    learn.save(save_name)
    num_rows = min(learn.data.batch_size, 3)
    learn.show_results(rows=num_rows, imgsize=5)

# loading 3 rounds of data
bs_1 = 64
size_1 = 128
db_1 = get_data(data_pth=data_pth, lr_dir=lr_path, hr_dir=hr_path,
             bs=bs_1, size=size_1)

bs_2 = 16
size_2 = 256
db_2 = get_data(data_pth=data_pth, lr_dir=lr_path, hr_dir=hr_path,
             bs=bs_2, size=size_2)

bs_3 = 8
size_3 = 512
db_3 = get_data(data_pth=data_pth, lr_dir=lr_path, hr_dir=hr_path,
             bs=bs_3, size=size_3)

# fitting round 1
learn = unet_learner(db_1, arch, 
                     wd=wd, 
                     #loss_func=feat_loss,
                     loss_func=F.mse_loss,
                     metrics=superres_metrics, 
                     #callback_fns=LossMetrics, 
                     blur=True, norm_type=NormType.Weight, model_dir=model_path)
do_fit(f'{nb_name}.1a', lr, cycle_len=1)
learn.unfreeze()
do_fit(f'{nb_name}.1b', slice(1e-5,lr), cycle_len=1)

# fitting round 2
learn = unet_learner(db_2, arch, 
                     wd=wd, 
                     #loss_func=feat_loss,
                     loss_func=F.mse_loss,
                     metrics=superres_metrics, 
                     #callback_fns=LossMetrics, 
                     blur=True, norm_type=NormType.Weight, model_dir=model_path)
learn.load(f'{nb_name}.1b')
do_fit(f'{nb_name}.2a', lr, cycle_len=3)
learn.unfreeze()
do_fit(f'{nb_name}.2b', slice(1e-5, lr), cycle_len=3)

# fitting round 3
learn = unet_learner(db_3, arch, 
                     wd=wd, 
                     #loss_func=feat_loss,
                     loss_func=F.mse_loss,
                     metrics=superres_metrics, 
                     #callback_fns=LossMetrics, 
                     blur=True, norm_type=NormType.Weight, model_dir=model_path)
learn.load(f'{nb_name}.2b')
do_fit(f'{nb_name}.3a', lr, cycle_len=3)
learn.unfreeze()
do_fit(f'{nb_name}.3b', slice(1e-5, lr/10), cycle_len=3)