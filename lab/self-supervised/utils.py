import numpy as np
from torchvision import transforms
from custom_transforms import RandomCutOut, ToGrayScale
import matplotlib.pyplot as plt
import math
from fastprogress.fastprogress import progress_bar
from fastai.callback import annealing_exp, Scheduler


def get_pssr_transforms(size=256, config=None):
    tfms = {}
    if not config:
        config = {'y_channel': 1, 'x_channel': 3}
    tfms['x'] = transforms.Compose([ToGrayScale(config['x_channel']), 
                                    transforms.Resize((size, int(size*1.5))),
                                    transforms.CenterCrop(size),
                                    transforms.ToTensor()])
    tfms['y'] = transforms.Compose([ToGrayScale(config['y_channel']), 
                                    transforms.Resize((size, int(size*1.5))),
                                    transforms.CenterCrop(size),
                                    transforms.ToTensor()])
    return tfms

def get_inpainting_transforms(size=(256,256), config=None):
    """Get transformations for Input and Target images"""
    tfms = {}
    # Sequence of augmentations for input images
    tfms['x'] = transforms.Compose([ToGrayScale(3), 
                                    transforms.CenterCrop(size),
                                    RandomCutOut(config['min_n_holes'], config['max_n_holes'], 
                                                 config['min_size'], config['max_size']),
                                    transforms.ToTensor()])
    # Sequence of augmentations for target images
    tfms['y'] = transforms.Compose([ToGrayScale(), 
                                    transforms.CenterCrop(size),
                                    transforms.ToTensor()])
    return tfms


def show_sample(dataset, idx=None, figsize=(20,20), seed=None):
    """A helper function to visualize data samples."""
    np.random.seed(seed=seed)
    if idx==None: 
        idx = np.random.randint(low=0, high=len(dataset)-1)
    x, y = dataset[idx]
    f, axarr = plt.subplots(1,2, figsize=figsize)  # create visualizations
    
    axarr[0].imshow(x.permute(1,2,0)) # visualize image tensor
    axarr[0].set_title('Input')
    axarr[1].imshow(y.permute(1,2,0).squeeze(), cmap=plt.cm.gray) # visualize image tensor
    axarr[1].set_title('Target')
    
    
def show_result(x, y, pred, figsize=(20,20)):
    """A helper function to visualize inference results."""
    f, axarr = plt.subplots(1,3, figsize=figsize)  # create visualizations
    axarr[0].imshow(x.permute(1,2,0)) # visualize image tensor
    axarr[0].set_title('Input')
    axarr[1].imshow(y.permute(1,2,0).squeeze(), cmap=plt.cm.gray) # visualize image tensor
    axarr[1].set_title('Target')
    axarr[2].imshow(pred[0].permute(1,2,0).squeeze(), cmap=plt.cm.gray) # visualize image tensor
    axarr[2].set_title('Prediction')
    
def find_lr(model, trn_loader, optimizer, loss_function, init_value = 1e-8, final_value=10., beta = 0.98):
    num = len(trn_loader)-1
    lr = init_value
    optimizer.param_groups[0]['lr'] = lr
    sched = Scheduler((init_value, final_value), 100, annealing_exp)
    avg_loss = 0.
    best_loss = 0.
    batch_num = 0
    losses = []
    log_lrs = []
    for data in progress_bar(trn_loader):
        batch_num += 1
        #As before, get the loss for this mini-batch of inputs/outputs
        inputs,labels = data
        inputs = inputs.cuda()
        labels = labels.cuda()
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_function(outputs, labels)
        #Compute the smoothed loss
        avg_loss = beta * avg_loss + (1-beta) *loss.data.item()
        smoothed_loss = avg_loss / (1 - beta**batch_num)
        #Stop if the loss is exploding
        if batch_num > 1 and smoothed_loss > 4 * best_loss:
            return log_lrs, losses
        #Record the best loss
        if smoothed_loss < best_loss or batch_num==1:
            best_loss = smoothed_loss
        #Store the values
        losses.append(smoothed_loss)
        log_lrs.append(math.log10(lr))
        #Do the SGD step
        loss.backward()
        optimizer.step()
        #Update the lr for the next step
        if batch_num >=100:
            break
        lr = sched.step()
        optimizer.param_groups[0]['lr'] = lr
    return log_lrs, losses