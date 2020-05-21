__all__ = ['DataLoader']

import numpy as np
import pandas as pd
from itertools import permutations
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from PIL import Image
from pathlib import Path
# train_set = pd.read_csv('../train.csv')
# valid_set = pd.read_csv('../valid.csv')
# data_pth = Path('/home/alaa/Dropbox/BPHO Staff/USF/EM/training/trainsets/hr/')

class DataLoader(Dataset):
    def __init__(self, data_pth, img_size=256, patch_size=150):
#         self.img_list = list(data_pth.glob('*.tif'))
        self.img_list = data_pth.values.reshape(-1).tolist()
        self.permutations = list(permutations([0,1,2,3]))
        self.__resize = transforms.Compose([
            transforms.Resize(img_size, Image.BILINEAR),
            transforms.CenterCrop(img_size)
        ])
        self.croplist = [[0, 0, patch_size, patch_size],
                     [0, img_size-patch_size, patch_size, img_size],
                     [img_size-patch_size, 0, img_size, patch_size],
                     [img_size-patch_size, img_size-patch_size, img_size, img_size]]
        self.__augment_tile = transforms.Compose([
#             transforms.RandomCrop(patch_size),
            transforms.ToTensor(),
        ])
    
    def __len__(self):
        return len(self.img_list)
    
    def __getitem__(self, index):
        img_name = self.img_list[index]
        img = Image.open(img_name).convert('RGB')
        img = self.__resize(img)
        tiles = [None]*4
        
        for n in range(4):
            tile = img.crop(self.croplist[n])
            tile = self.__augment_tile(tile)
            # Normalize the patches indipendently to avoid low level features shortcut
            m, s = tile.mean(), tile.std()
            norm = transforms.Normalize(mean=m.view(1), std=s.view(1))
            tile = norm(tile)
            tiles[n] = tile

        order = np.random.randint(len(self.permutations))
        data = [tiles[self.permutations[order][t]] for t in range(4)]
        data = torch.stack(data, 0)
        self.x = data
        self.y = int(order)
        return self.x, self.y
    
# jigsaw_train = DataLoader(train_set)
# jigsaw_valid = DataLoader(valid_set)