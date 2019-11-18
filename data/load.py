from fastai import *
from fastai.vision import *
from fastai.callbacks import *


"""
Usage: bs = 8
       size = 128
       data_pth = Path('/home/alaa/Dropbox (BPHO)/BPHO Staff/USF')
       lr_dir = f'EM/training/trainsets/crappified/'
       hr_dir = f'EM/training/trainsets/hr/'
       db = get_data(data_pth, lr_dir, hr_dir, bs, size)
"""


def get_src(data_pth, lr_dir, hr_dir):
    hr_tifs = data_pth/f'{hr_dir}'
    lr_tifs = data_pth/f'{lr_dir}'

    def map_to_hr(x):
        hr_name = x.relative_to(lr_tifs)
        return hr_tifs/hr_name
    print(lr_tifs)
    src = (ImageImageList
            .from_folder(lr_tifs)
            .split_by_rand_pct()
            .label_from_func(map_to_hr))
    return src


def get_data(data_pth, lr_dir, hr_dir, bs, size,
             num_workers=4, noise=None, max_zoom=1.1):
    src = get_src(data_pth, lr_dir, hr_dir)
    tfms = get_transforms(flip_vert=True, max_zoom=max_zoom)
    data = (src
            .transform(tfms, size=size)
            .transform_y(tfms, size=size)
            .databunch(bs=bs, num_workers=num_workers)
            .normalize(imagenet_stats, do_y=True))
    data.c = 3
    return data
