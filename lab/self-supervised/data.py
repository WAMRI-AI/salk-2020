import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import numpy as np

class EMPSSR(Dataset):
    """A PyTorch Dataset class for self-supervised learning
    on Electro-magnetic Microscopy Images.

    :param data_pth: Path object containing the absolute path to the target images
    :param transforms: Dictionary of transformations for the input and target images.

    The transforms for the inputs defines the self-supervised pre-text task.
    See get_selfplay_transforms() in utils.py for an example definition.
    """

    def __init__(self, target_pth, transforms=None):
        self.target_img_filepaths = np.squeeze(target_pth).tolist()
        self.transforms = transforms
        self.c = int(str(transforms['y']).split()[1][-2])

    def __len__(self):
        return len(self.target_img_filepaths)

    def __getitem__(self, idx):
#         if torch.is_tensor(idx):
#             idx = idx.tolist()
        input_file = self.target_img_filepaths[idx].replace('hr', 'lr')
        target_file = self.target_img_filepaths[idx]
        x = Image.open(input_file)
        y = Image.open(target_file)
        if self.transforms:
            x = self.transforms['x'](x)
            y = self.transforms['y'](y)
        return x, y

class EMSelfPlay(Dataset):
    """A PyTorch Dataset class for self-supervised learning 
    on Electro-magnetic Microscopy Images.
    
    :param data_pth: Path object containing the absolute path to the target images
    :param transforms: Dictionary of transformations for the input and target images. 
    
    The transforms for the inputs defines the self-supervised pre-text task.
    See get_selfplay_transforms() in utils.py for an example definition.
    """

    def __init__(self, data_pth, transforms=None):
        self.img_filepaths = np.squeeze(data_pth).tolist()
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


