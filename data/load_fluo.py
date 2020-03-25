from fastai import *
from fastai.vision import *
from fastai.callbacks import *
import sys
sys.path.append('../')
from model.bpho.multi import MultiImageImageList

def get_src(x_data, y_data, n_frames=1, mode='L'):
    def map_to_hr(x):
        new_path = x.relative_to(x_data).with_suffix('.tif')
        new_path = str(new_path).replace('lr', 'hr')
        return y_data/new_path

    if n_frames == 1:
        src = (ImageImageList
                .from_folder(x_data, convert_mode=mode)
                .split_by_folder()
                .label_from_func(map_to_hr, convert_mode=mode))
    else:
        src = (MultiImageImageList
                .from_folder(x_data, extensions=['.npy'])
                .split_by_folder()
                .label_from_func(map_to_hr, convert_mode=mode))
    return src

def get_data(bs, size, x_data, y_data,
             n_frames=1,
             max_rotate=10.,
             min_zoom=1., max_zoom=1.1,
             use_cutout=False,
             use_noise=True,
             scale=4,
             xtra_tfms=None,
             gauss_sigma=(0.4,0.7),
             pscale=(5,30),
             mode='L',
             norm=False,
             **kwargs):
    src = get_src(x_data, y_data, n_frames=n_frames, mode=mode)
    x_tfms, y_tfms = get_xy_transforms(
                          max_rotate=max_rotate,
                          min_zoom=min_zoom, max_zoom=max_zoom,
                          use_cutout=use_cutout,
                          use_noise=use_noise,
                          gauss_sigma=gauss_sigma,
                          pscale=pscale,
                          xtra_tfms = xtra_tfms)
    x_size = size // scale
    data = (src
            .transform(x_tfms, size=x_size)
            .transform_y(y_tfms, size=size)
            .databunch(bs=bs, **kwargs))
    if norm:
        print('normalizing x and y data')
        data = data.normalize(do_y=True)
    data.c = 1
    return data 

def get_patched_src(y_data, n_frames=1, mode='L'):
#     if n_frames == 1:
    src = (ImageImageList
            .from_folder(y_data, convert_mode=mode)
            .split_by_folder()
            .label_from_func(lambda x: x, convert_mode=mode))
#     else:
#         src = (MultiImageImageList
#                 .from_folder(y_data)
#                 .split_by_folder()
#                 .label_from_func(lambda x: x, convert_mode=mode))
    return src
 

def get_patched_data(bs, size, x_data, y_data,
             n_frames=1, num_workers=4, tfms=None, mode='L'):
    src = get_patched_src(y_data, n_frames=n_frames, mode=mode)
    data = (src
                .transform(tfms, size=size, resize_method=ResizeMethod.CROP, tfm_y=False)
                .transform_y(None, size=size, resize_method=ResizeMethod.CROP)
                .databunch(bs=bs, num_workers=num_workers)
                .normalize(do_y=True))
    data.c = 1
    data.train_ds.tfms_y = None
    data.valid_ds.tfms_y = None
    return data

def get_xy_transforms(max_rotate=10., min_zoom=1., max_zoom=2., use_cutout=False, use_noise=False, xtra_tfms = None,
                      gauss_sigma=(0.01,0.05), pscale=(5,30)):
    base_tfms = [[
            rand_crop(),
            dihedral_affine(),
            rotate(degrees=(-max_rotate,max_rotate)),
            rand_zoom(min_zoom, max_zoom)
        ],
        [crop_pad()]]

    y_tfms = [[tfm for tfm in base_tfms[0]], [tfm for tfm in base_tfms[1]]]
    x_tfms = [[tfm for tfm in base_tfms[0]], [tfm for tfm in base_tfms[1]]]
    if use_cutout: x_tfms[0].append(cutout(n_holes=(5,10)))
    if use_noise:
        x_tfms[0].append(my_noise(gauss_sigma=gauss_sigma, pscale=pscale))
        #x_tfms[1].append(my_noise(gauss_sigma=(0.01,0.05),pscale=(5,30)))

    if xtra_tfms:
        for tfm in xtra_tfms:
            x_tfms[0].append(tfm)

    return x_tfms, y_tfms