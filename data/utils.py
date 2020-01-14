import numpy as np
import fastai.vision as faiv
import scipy.ndimage
import libtiff
import imageio

def load_img(img_filename):
    """Loads input image into matrix using filename"""
    img = libtiff.TiffFile(img_filename)
    img_mat = img.get_tiff_array()[0].astype(np.float32)[np.newaxis, :]
    return img_mat

def save_img(img, filepath, format="tiff"):
    """Saves input matrix into image file using filepath"""
    img = img_to_uint8(img)
    imageio.mimwrite(filepath, img)

def img_to_uint8(img):
    """Converts input matrix into 8-bit"""
    return img.astype(np.uint8)

def crop_center(img, x, y):
    c, h, w = img.shape
    crop_x = (w - x) // 2
    crop_y = (h - y) // 2    
    return img[:, crop_y:h-crop_y, crop_x:w-crop_x]

def bilinear_upsample(img, scale=4):
    img_upsampled = scipy.ndimage.zoom(np.squeeze(img), 4, order=1)
    return np.expand_dims(img_upsampled, axis=0)

def _custom_cutout(x, min_n_holes:faiv.uniform_int=5, max_n_holes:faiv.uniform_int=10,
                   min_length:faiv.uniform_int=5, max_length:faiv.uniform_int=15):
    "Cut out `n_holes` number of square holes of size `length` in image at random locations."
    h,w = x.shape[1:]
    n_holes = np.random.randint(min_n_holes, max_n_holes)
    h_length = np.random.randint(min_length, max_length)
    w_length = np.random.randint(min_length, max_length)
    for n in range(n_holes):
        h_y = np.random.randint(0, h)
        h_x = np.random.randint(0, w)
        y1 = int(np.clip(h_y - h_length / 2, 0, h))
        y2 = int(np.clip(h_y + h_length / 2, 0, h))
        x1 = int(np.clip(h_x - w_length / 2, 0, w))
        x2 = int(np.clip(h_x + w_length / 2, 0, w))
        x[:, y1:y2, x1:x2] = 0
    return x

custom_cutout = faiv.TfmPixel(_custom_cutout, order=20)