import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import numpy as np



class EMSelfPlay(Dataset):
    """A PyTorch Dataset class for self-supervised learning 
    on Electro-magnetic Microscopy Images.
    
    :param data_pth: Path object containing the absolute path to the target images
    :param transforms: Dictionary of transformations for the input and target images. 
    
    The transforms for the inputs defines the self-supervised pre-text task.
    See get_selfplay_transforms() in utils.py for an example definition.
    """

    def __init__(self, data_pth, transforms=None):
        self.img_filepaths = list(data_pth.glob('*.tif'))
        self.transforms = transforms
        
    def __len__(self):
        return len(self.img_filepaths)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        img_name = self.img_filepaths[idx]
        y = Image.open(img_name)
        if self.transforms:
            x = self.transforms['x'](y)
            y = self.transforms['y'](y)
        else:
            x = y
            print('WARNING: no transforms applied to target images, are you sure about that bud?')
        return x, y


