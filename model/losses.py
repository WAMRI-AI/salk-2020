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
def flatten_model(model):
    """Using children() method, flatten a complex model."""
    flattened = []

    def get_children(block):
        for child in list(block.children()):
            grand_children = list(child.children())
            if len(grand_children):
                get_children(child)
            else: flattened.append(child)
    
    get_children(model)
    return flattened

def find_layers(flattened_model):
    """Find the layers previous to the grid-changing layers in a flattened model."""
    
    def is_grid_changing(layer):
        """add controls here"""
        if 'pooling' in str(type(layer)): return True
        if isinstance(layer, torch.nn.modules.conv.Conv2d) and layer.stride==(2,2):
            return True
    
    loss_features = []
    for i, layer in enumerate(flattened_model[1:]):
        if is_grid_changing(layer):
            loss_features.append(flattened_model[i]) 
            # append the layer previous to the grid-changing ones
            # want to see the grid-changing ones? add the index by 1
            # loss_features.append(flattened_model[i+1]) 
    return loss_features

class FeatureLoss(nn.Module):
    def __init__(self, m_feat, layer_wgts):
        super().__init__()
        self.__name__ = 'feat_loss'
        self.m_feat = m_feat
        self.loss_features = find_layers(flatten_model(self.m_feat))
        self.hooks = hook_outputs(self.loss_features, detach=False)
        self.wgts = layer_wgts
        self.metric_names = ['pixel',] + [
            f'feat_{i}' for i in range(len(self.loss_features))
              ] + [f'gram_{i}' for i in range(len(self.loss_features))]
    
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