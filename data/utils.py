import numpy as np
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