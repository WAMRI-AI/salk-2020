# training param
train_date = '5.4'
gpu_id = 2
sample = False
pretrained = True
import torch.nn.functional as F
loss_function = F.mse_loss
config = {'y_channel': 3, 'x_channel': 3}

from enum import Enum
from pathlib import Path
import sys
sys.path.append('../')
sys.path.append('../../../model')

from metrics import *
from losses import flatten_model
from data import EMPSSR
from custom_transforms import RandomCutOut, GaussianBlur, ToGrayScale
from utils import show_sample, show_result, get_pssr_transforms, find_lr
from model import DynamicUnet

import torch
from torch import nn

from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def get_lrs(dataloader, max_lr=1e-3, min_lr=None, max_mom=0.95):
    """This function returns arrays for learning rates and momentum
    following the One Cycle Policy"""
    num_batches = len(dataloader)
    if not min_lr: min_lr = max_lr/10
    lrs_up = np.linspace(min_lr, max_lr, num_batches // 2)
    lrs_down = np.linspace(max_lr, min_lr, (num_batches // 2)+1)

    moms_up = np.linspace(max_mom-0.1, max_mom, num_batches // 2)
    moms_down = np.linspace(max_mom, max_mom-0.1, (num_batches // 2)+1)
    if not num_batches%2:
        lrs_down = lrs_down[1:]
        moms_down = moms_down[:-1]
    total_lrs = np.concatenate([lrs_up, lrs_down])
    total_moms = np.concatenate([moms_down, moms_up])
    return total_lrs, total_moms

def train(num_epochs):
    total_loss = 0.0
    print_every = 10
    running_loss = 0.0
    model.cuda()
    for i in range(num_epochs):
        model.train()
        print("Training Model...")
        for j, sample_batch in enumerate(train_dl):
            # get the inputs; data is a list of [inputs, labels]
            x, y = sample_batch
            x, y = x.cuda(), y.cuda()

            # zero the parameter gradients
            optimizer.zero_grad()

            preds = model(x)  # [N,C]

            loss = loss_function(y, preds)
            loss.backward()
            optimizer.step()

            # print statistics
            total_loss += loss.item()
            running_loss += loss.item()
            if j % (print_every) == 0:
                print(f'Epoch {i+1} Batch {j+1} loss: {running_loss/print_every}')
                running_loss = 0.0

            # update optimizer params
            optimizer.param_groups[0]['lr'] = lrs[j]
            optimizer.param_groups[0]['momentum'] = moms[j]

        model.eval()
        train_loss = total_loss / (j+1)
        print("Validating Model...")
        valid_dict = validate(model)
        print(f'Epoch {i+1}, Train loss: {train_loss:.4f},\nValid loss:{valid_dict["valid_loss"]:.4f},\nValid psnr: {valid_dict["valid_psnr"]:.2f},\nValid ssim: {valid_dict["valid_ssim"]:.4f}')

def validate(model):
    loss_total = 0.0
    psnr_total = 0.0
    ssim_total = 0.0
    len_ds = 0
    for i, sample_batch in enumerate(valid_dl):
        x, y = sample_batch
        len_ds += len(y)
        x, y = x.cuda(), y.cuda()
        preds = model(x)  # [N,C]

        loss_batch = loss_function(y, preds).item()
        loss_total += loss_batch * len(y)
        for pred, target in zip(preds, y):
            psnr_per_image = psnr(pred, target).item()
            psnr_total += psnr_per_image

        ssim_avg = ssim(preds, y).item()
        ssim_total += ssim_avg * len(y)

    return {'valid_loss': loss_total / len_ds,
            'valid_psnr': psnr_total / len_ds,
            'valid_ssim': ssim_total / len_ds,
           }

def save(model_name):
    save_pth = model_pth/model_kind/(model_name+'.pkl')
    torch.save(model.state_dict(), save_pth)
    print(f'Model saved: {model_name}')

def freeze():
    for layer in flattened[:90]:
        for p in layer.parameters():
            p.requires_grad = False

def unfreeze():
    for layer in flattened[:90]:
        for p in layer.parameters():
            p.requires_grad = True

def get_loaders(bs, size, sample=False):
    train_set = pd.read_csv('../train.csv')
    valid_set = pd.read_csv('../valid.csv')
    if sample:
        train_set = train_set.sample(frac=0.1)
        valid_set = valid_set.sample(frac=0.1)
    tfms = get_pssr_transforms(size, config)
    train_ds = EMPSSR(train_set, tfms)
    valid_ds = EMPSSR(valid_set, tfms)
    train_dl = DataLoader(train_ds, batch_size=bs,
                          shuffle=True, num_workers=4)
    valid_dl = DataLoader(valid_ds, batch_size=bs,
                          shuffle=True, num_workers=4)
    return train_dl, valid_dl
              

# SPECS
num_cores = 4
torch.cuda.set_device(gpu_id)
train_set = pd.read_csv('../train.csv')
valid_set = pd.read_csv('../valid.csv')
arch = models.resnet34(pretrained)
encoder = nn.Sequential(*list(arch.children())[:-2])
nt = Enum('NormType', 'Batch BatchZero Weight Spectral')
model_kind = 'baseline/'
model_pth = Path('/home/alaa/Dropbox/BPHO Staff/USF/EM/models/self_sv/')

# Round 1
bs = 64  # batch size
size = 128  # image size
train_dl, valid_dl = get_loaders(bs, size, sample=sample)
model = DynamicUnet(encoder, n_classes=config['y_channel'],
                    img_size=(size,size), norm_type=nt.Weight)
flattened = flatten_model(model)
## 1a
lrs, moms = get_lrs(train_dl, max_lr=1e-3)
optimizer = torch.optim.SGD(model.parameters(), lr=lrs[0], momentum=moms[0])
train(1)
## 1b
lrs, moms = get_lrs(train_dl, max_lr=1e-3, min_lr=1e-5)
optimizer = torch.optim.SGD(model.parameters(), lr=lrs[0], momentum=moms[0])
train(1)
##save&export
model_name = train_date + '_round_1'
save(model_name)

# Round 2
bs = 16  # batch size
size = 256  # image size
train_dl, valid_dl = get_loaders(bs, size, sample=sample)
encoder.cpu()
model = DynamicUnet(encoder, n_classes=config['y_channel'],
                    img_size=(size,size), norm_type=nt.Weight)
model.cuda()
model_name = train_date + '_round_1'
save_pth = model_pth/model_kind/(model_name+'.pkl')
model.load_state_dict(torch.load(save_pth))
## 2a
flattened = flatten_model(model)
freeze()
lrs, moms = get_lrs(train_dl, max_lr=1e-3)
optimizer = torch.optim.SGD(model.parameters(), lr=lrs[0], momentum=moms[0])
train(3)
## 2b
unfreeze()
lrs, moms = get_lrs(train_dl, max_lr=1e-3, min_lr=1e-5)
optimizer = torch.optim.SGD(model.parameters(), lr=lrs[0], momentum=moms[0])
train(3)
##save&export
model_name = train_date + '_round_2'
save(model_name)

# Round 3
bs = 8  # batch size
size = 512  # image size
train_dl, valid_dl = get_loaders(bs, size, sample=sample)
encoder.cpu()
model = DynamicUnet(encoder, n_classes=config['y_channel'],
                    img_size=(size,size), norm_type=nt.Weight)
model.cuda()
model_name = train_date + '_round_2'
save_pth = model_pth/model_kind/(model_name+'.pkl')
model.load_state_dict(torch.load(save_pth))
## 3a
flattened = flatten_model(model)
freeze()
lrs, moms = get_lrs(train_dl, max_lr=1e-3)
optimizer = torch.optim.SGD(model.parameters(), lr=lrs[0], momentum=moms[0])
train(3)
## 3b
unfreeze()
lrs, moms = get_lrs(train_dl, max_lr=1e-4, min_lr=1e-5)
optimizer = torch.optim.SGD(model.parameters(), lr=lrs[0], momentum=moms[0])
train(3)
##save&export
model_name = train_date + '_round_3'
save(model_name)
