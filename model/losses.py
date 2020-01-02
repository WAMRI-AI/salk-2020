import torch
import torch.nn as nn
import torch.nn.functional as F


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