import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from PIL import Image
class DataLoader(Dataset):
    def __init__(self, data_pth):
        self.img_list = list(data_pth.glob('*.tif'))

        self.permutations = self.__retrive_permutations()

        self.__image_transformer = transforms.Compose([
            transforms.Resize(256, Image.BILINEAR),
            transforms.CenterCrop(255)])
        self.__augment_tile = transforms.Compose([
            transforms.RandomCrop(64),
            transforms.Resize((75, 75), Image.BILINEAR),
            transforms.ToTensor(),
        ])
    
    def __len__(self):
        return len(self.img_list)
    
    def __getitem__(self, index):
        img_name = self.img_list[index]
        img = Image.open(img_name).convert('L')
        img = self.__image_transformer(img)

        s = float(img.size[0]) / 3
        a = s / 2
        tiles = [None] * 9
        for n in range(9):
            i = n / 3
            j = n % 3
            c = [a * i * 2 + a, a * j * 2 + a]
            c = np.array([c[1] - a, c[0] - a, c[1] + a + 1, c[0] + a + 1]).astype(int)
            tile = img.crop(c.tolist())
            tile = self.__augment_tile(tile)
            # Normalize the patches indipendently to avoid low level features shortcut
            m, s = tile.mean(), tile.std()
            norm = transforms.Normalize(mean=m.view(1), std=s.view(1))
            tile = norm(tile)
            tiles[n] = tile

        order = np.random.randint(len(self.permutations))
        data = [tiles[self.permutations[order][t]] for t in range(9)]
        data = torch.stack(data, 0)
        self.x = data
        self.y = int(order)
        return self.x, self.y

    def __retrive_permutations(self):
        all_perm = np.load('permutations_1000.npy')
        return all_perm