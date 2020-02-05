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


def get_data(data_pth, lr_dir, hr_dir, bs, in_sz, out_sz,
             num_workers=4, noise=None, max_zoom=1.1):
    src = get_src(data_pth, lr_dir, hr_dir)
    tfms = get_transforms(flip_vert=True, max_zoom=max_zoom)
    data = (src
            .transform(tfms, size=in_sz, resize_method=ResizeMethod.CROP)
            .transform_y(tfms, size=out_sz, resize_method=ResizeMethod.CROP)
            .databunch(bs=bs, num_workers=num_workers)
            .normalize(imagenet_stats, do_y=True))
    data.c = 3
    return data

def get_patched_data(data_pth, hr_dir, bs, in_sz, out_sz,
                     tfms=None, num_workers=4, noise=None, max_zoom=1.1):
    src = get_patched_src(data_pth, hr_dir)
    
    data = (src
                .transform(tfms, size=in_sz, resize_method=ResizeMethod.CROP, tfm_y=False)
                .transform_y(None, size=out_sz, resize_method=ResizeMethod.CROP)
                .databunch(bs=bs, num_workers=num_workers)
                .normalize(imagenet_stats, do_y=True))
    data.c = 3
    data.train_ds.tfms_y = None
    data.valid_ds.tfms_y = None
    return data


def get_patched_src(data_pth, hr_dir):
    hr_tifs = data_pth/f'{hr_dir}'

    def map_to_hr(x):
        return hr_tifs/x.relative_to(hr_tifs).with_suffix(".tif")

    src = (ImageImageList
            .from_folder(hr_tifs)
            .split_by_rand_pct()
            .label_from_func(map_to_hr))
    return src

def subsample(data, pct=0.1, seed=None):
    """Takes a databunch as input and returns a mini-version of the dataset
    This is useful for debugging and rapid experimentation. 
    data -> a databunch object
    pct  -> the fraction of original dataset size (default: 0.1)"""
    if seed: np.random.seed(seed)
    trn_size = len(data.train_ds)
    trn_indices = np.random.choice(np.arange(trn_size), 
                                   size=int(pct*trn_size), replace=False)
    trn_sampler = torch.utils.data.sampler.SubsetRandomSampler(trn_indices)
    val_size = len(data.valid_ds)
    val_indices = np.random.choice(np.arange(val_size), size=int(pct*val_size), replace=False)
    val_sampler = torch.utils.data.sampler.SubsetRandomSampler(val_indices)
    mini_data = copy(data)
    mini_data.train_dl = mini_data.train_dl.new(shuffle=False, sampler=trn_sampler)
    mini_data.valid_dl = mini_data.valid_dl.new(shuffle=False, sampler=val_sampler)
    return mini_data

def get_test(lr_files, data_pth, lr_dir, hr_dir, bs, in_sz, out_sz,
             num_workers=4, noise=None, max_zoom=1.):
    src = get_src(data_pth, lr_dir, hr_dir)
    data = (src
            .add_test(lr_files)
            .databunch(bs=bs, num_workers=num_workers)
            .normalize(imagenet_stats, do_y=True))
    data.c = 3
    return data

def get_src(data_pth, lr_dir, hr_dir, seed=49):
    hr_tifs = data_pth/f'{hr_dir}'
    lr_tifs = data_pth/f'{lr_dir}'

    def map_to_hr(x):
        return Path(str(hr_tifs/x.relative_to(lr_tifs).with_suffix(".tif")).replace('lr', 'hr'))

    src = (ImageImageList
            .from_folder(lr_tifs)
            .split_by_rand_pct(seed=seed)
            .label_from_func(map_to_hr))
    return src