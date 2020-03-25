import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from math import exp
from fastai import *
from fastai.vision import *
from fastai.callbacks import *


### MSE Loss
mse_loss = F.mse_loss

### Base Loss 
base_loss = F.l1_loss

### Gram Loss
def gram_matrix(x):
    n,c,h,w = x.size()
    x = x.view(n, c, -1)
    return (x @ x.transpose(1,2))/(c*h*w)

### Feature Loss 
class FeatureLoss(nn.Module):
    def __init__(self, m_feat, layer_ids, layer_wgts):
        super().__init__()
        self.__name__ = 'feat_loss'
        self.m_feat = m_feat
        self.loss_features = self.make_layers(layer_ids)
        self.hooks = hook_outputs(self.loss_features, detach=False)
        self.wgts = layer_wgts
        self.metric_names = ['pixel',] + [f'feat_{i}' for i in range(len(layer_ids))
              ] + [f'gram_{i}' for i in range(len(layer_ids))]

    def make_layers(self, layer_ids):
        loss_features = []
        for layer in layer_ids:
            obj = self.m_feat
            for i in layer:
                try:
                    obj = obj[i]
                except TypeError:
                    children = list(obj.children())
                    if len(children):
                        obj = children[i]
                    else:
                        break
            loss_features.append(obj)
        return loss_features
    
    def make_features(self, x, clone=False):
        self.m_feat(x)
        return [(o.clone() if clone else o) for o in self.hooks.stored]
    
    def forward(self, pred, target):
        out_feat = self.make_features(target, clone=True)
        in_feat = self.make_features(pred)
        self.feat_losses = [base_loss(pred,target)]
        self.feat_losses += [base_loss(f_in, f_out)*w
                             for f_in, f_out, w in zip(in_feat, out_feat, self.wgts)]
        self.feat_losses += [base_loss(gram_matrix(f_in), gram_matrix(f_out))*w**2 * 5e3
                             for f_in, f_out, w in zip(in_feat, out_feat, self.wgts)]
        self.metrics = dict(zip(self.metric_names, self.feat_losses))
        return sum(self.feat_losses)

    def __del__(self): self.hooks.remove()


def grab_resnet_layers(resnet):
    """returns layers that have Conv2D with stride = 2 and indices
    of layers just before each of them (if one exists)"""
    layers, layer_ind = pooling_layer(resnet)
    act_blocks = previous_layer(layer_ind)
    return layers, act_blocks

def next_ind(cur_ind, i):
    ind = cur_ind.copy()
    ind.append(i)
    return ind

def pooling_layer(block, cur_ind=[]):
    """cur_ind: the index of Sequential or BasicBlock that the 
    code is looking into"""
    returned_block = []
    returned_index = []

    for i, k in enumerate(block):
        if isinstance(k, (torch.nn.modules.container.Sequential, torchvision.models.resnet.BasicBlock)):
            new_block, new_index = pooling_layer(k.children(), next_ind(cur_ind, i))
            returned_block.extend(new_block)
            returned_index.extend(new_index)
        elif isinstance(k, torch.nn.modules.conv.Conv2d) and k.stride==(2,2):
            returned_block.append(k)
            returned_index.append(next_ind(cur_ind, i))
    return returned_block, returned_index

def previous_layer(layer_ind):
    prev_layer_list = []
    for ind in layer_ind:
        if not sum(ind):
            continue
        prev_layer = ind.copy()
        prev_layer[-1] -= 1
        if prev_layer[-1] == -1:
            for i in list(range(len(prev_layer)))[::-1]:
                if prev_layer[i] == -1:
                    prev_layer[i-1] -= 1
        prev_layer_list.append(prev_layer)
    return prev_layer_list


### MSE Loss with Sharpness Regularization
def sharp_loss(output, target, alpha=0.0001):
    # Compute MSE term using output and target images
    mse_term = F.mse_loss(output, target)
    # compute sharpness term using L2 norm of output image gradient
    sharp_term = torch.norm(img_grad(output), p=2)
    # compute overall Sharp Loss
    return mse_term + (alpha*sharp_term)


def img_grad(x, num_channels=3):
    # Create Localized horizontal gradient convolution kernel (Sobel)
    h = torch.Tensor([[1, 0, -1],
                      [2, 0, -2],
                      [1, 0, -1]])
    # Expand dims to account for (batch_size, num_channels, height, width)
    h = h.expand(num_channels, -1, -1).view((1, num_channels, 3, 3)).cuda()
    # Apply conv layers using Sobel kernel
    G_x = F.conv2d(x, h, padding=1, stride=1, bias=None)
    G_y = F.conv2d(x, h.T.view((1,num_channels,3,3)), padding=1, stride=1, bias=None)
    # Compute image gradient using Finite Difference Approximation
    G = torch.sqrt(torch.pow(G_x, 2) + torch.pow(G_y, 2))
    return G

### SSIM Loss
def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    return gauss/gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def _ssim(img1, img2, window, window_size, channel, size_average = True):
    mu1 = F.conv2d(img1, window, padding = window_size//2, groups = channel)
    mu2 = F.conv2d(img2, window, padding = window_size//2, groups = channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1*mu2

    sigma1_sq = F.conv2d(img1*img1, window, padding = window_size//2, groups = channel) - mu1_sq
    sigma2_sq = F.conv2d(img2*img2, window, padding = window_size//2, groups = channel) - mu2_sq
    sigma12 = F.conv2d(img1*img2, window, padding = window_size//2, groups = channel) - mu1_mu2

    C1 = 0.01**2
    C2 = 0.03**2

    ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)

class SSIM(torch.nn.Module):
    def __init__(self, window_size = 11, size_average = True):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = create_window(window_size, self.channel)

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()

        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = create_window(self.window_size, channel)
            
            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            window = window.type_as(img1)
            
            self.window = window
            self.channel = channel


        return _ssim(img1, img2, window, self.window_size, channel, self.size_average)

def ssim_loss(img1, img2, window_size = 11, size_average = True):
    (_, channel, _, _) = img1.size()
    window = create_window(window_size, channel)
    
    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)
    
    return -1.*_ssim(img1, img2, window, window_size, channel, size_average)

def ssim_log_loss(img1, img2, window_size = 11, size_average = True):
    (_, channel, _, _) = img1.size()
    window = create_window(window_size, channel)
    
    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)
    
    return -1.*torch.log(_ssim(img1, img2, window, window_size, channel, size_average))