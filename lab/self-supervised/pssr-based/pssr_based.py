import argparse

parser = argparse.ArgumentParser(description='Get the critic.')
parser.add_argument('critic', metavar='N', type=str, 
                    help='which critic will we use? i = inpainting, c = contrastive, s = self-critic')
parser.add_argument('--gpu', type=int, 
                    help='gpu id')
parser.add_argument('-d', type=str, 
                    help='train_date')

args = parser.parse_args()
critic_type = args.critic
gpu_id = args.gpu
train_date = args.d

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
from clr import *

# Basic Setup
torch.cuda.set_device(gpu_id)
data_pth = Path('/home/alaa/Dropbox/BPHO Staff/USF/')
lr_path = f'EM/training/trainsets/lr/'
hr_path = f'EM/training/trainsets/hr/'
model_pth = data_pth/f'EM/models/'

# Resnet Feature loss

bs = 8
size = 512
superres_metrics = [F.mse_loss, psnr, ssim]

def inpaint_data(bs, size):
    random_patch = partial(custom_cutout, min_n_holes=10, max_n_holes=20,
                           min_length=15, max_length=25, use_on_y=False)
    tfms = [[random_patch()], [random_patch()]]
    data = get_patched_data(data_pth, hr_path, bs, tfms=tfms,
                            in_sz=size, out_sz=size)
    return data

def clr_data(bs, size):
    random_patch = partial(custom_cutout, min_n_holes=10, max_n_holes=20,
                       min_length=5, max_length=10, use_on_y=False)
    patch_tfms = [random_patch()]
    # Create databunch
    data = get_clr_data(data_pth, hr_dir=hr_path, bs=bs, xtra_tfms=patch_tfms,
                        in_sz=size, out_sz=size)
    return data

def pssr_data(bs, size):
    return get_data(data_pth=data_pth, lr_dir=lr_path, hr_dir=hr_path,
            bs=bs, in_sz=size, out_sz=size, max_zoom=2.)

base_model = 'baselines/emsynth_005_unet.5'
critic_dict = {
    'i': ['self_sv/inpainting/critic_inpaint_best', inpaint_data],
    's': [base_model, pssr_data]
}

if critic_type=='c':
    critic_data = clr_data(bs, size)
    critic = get_clr_learner(gpu_id, critic_data, model_pth)
    gc.collect()
    critic_model = 'self_sv/contrastive/critic-clr.4b'
    critic.load(critic_model)
    encoder = critic.model.encoder.eval().cuda()
else:
    critic_model, _critic_data = critic_dict[critic_type]
    critic_data = _critic_data(bs, size)
    critic_arch = models.resnet34
    wd = 1e-3
    critic = unet_learner(critic_data, critic_arch, wd=wd,
                             loss_func=F.mse_loss,
                             metrics=superres_metrics,
                             blur=True,
                             norm_type=NormType.Weight,
                             model_dir=model_pth)
    gc.collect()
    critic.load(critic_model)
    encoder = critic.model.eval().cuda()[0]

print(f'Critic loaded: {critic_model}')

feature_maps = losses.find_layers(flatten_model(encoder))
num_layers = len(feature_maps)
feat_loss = losses.FeatureLoss(m_feat=encoder, layer_wgts=[1/num_layers for _ in range(num_layers)])

# PSSR Loading

# Training
arch = models.resnet34
wd = 1e-3

learn_model, _learn_data = critic_dict['s']
learn_data = _learn_data(bs, size)
learn = unet_learner(learn_data, arch, wd=wd,
                                 loss_func=feat_loss,
                                 metrics=superres_metrics,
                                 blur=True,
                                 norm_type=NormType.Weight,
                                 model_dir=model_pth)
learn.load(learn_model)

learn.model_dir = model_pth/'self_sv/pssr_based'

lr = 1e-3

def do_fit(save_name, lrs=slice(lr), pct_start=0.9, cycle_len=10):
    learn.fit_one_cycle(cycle_len, lrs, pct_start=pct_start)
    learn.save(save_name)
    print(f'Model saved: {save_name}')
#     num_rows = min(learn.data.batch_size, 3)
#     learn.show_results(rows=num_rows, imgsize=5)

critic_names = {
    'c': 'contrastive',
    'i': 'inpaint',
    's': 'self-critic'
}

save_name = train_date + '_' + critic_names[critic_type]

do_fit(f'{save_name}.1a', lr, cycle_len=3)
learn.unfreeze()
do_fit(f'{save_name}.1b', slice(1e-5,lr/10), cycle_len=3)