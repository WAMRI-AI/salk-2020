import numpy as np
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