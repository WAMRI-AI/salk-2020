import numpy as np
from torchvision import transforms
from custom_transforms import RandomCutOut, ToGrayScale
import matplotlib.pyplot as plt



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
    axarr[2].imshow(pred.permute(3,2,0,1).squeeze(), cmap=plt.cm.gray) # visualize image tensor
    axarr[2].set_title('Prediction')