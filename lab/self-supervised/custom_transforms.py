from torchvision import transforms
import numpy as np
from PIL import Image
import cv2



class GaussianBlur(object):
    """Implements Gaussian blur as described in the SimCLR paper"""
    def __init__(self, kernel_size, min=0.1, max=2.0):
        self.min = min
        self.max = max
        # kernel size is set to be 10% of the image height/width
        self.kernel_size = kernel_size

    def __call__(self, sample):
        sample = np.array(sample)

        # blur the image with a 50% chance
        prob = np.random.random_sample()

        if prob < 0.5:
            sigma = (self.max - self.min) * np.random.random_sample() + self.min
            sample = cv2.GaussianBlur(sample, (self.kernel_size, self.kernel_size), sigma)

        return sample


class ToGrayScale(object):
    """Convert a ``PIL Image`` to gray-scale.
    
    :param num_outputs: int, default is 1, otherwise channels repeated. 

    In the other cases, tensors are returned without scaling.
    """
    def __init__(self, num_outputs=1):
        self.num_outputs = num_outputs
    

    def __call__(self, pic):
        """
        Args:
            pic (PIL Image or numpy.ndarray): Image to be converted to tensor.

        Returns:
            Tensor: Converted image.
        """
        return transforms.functional.to_grayscale(pic, self.num_outputs)


    def __repr__(self):
        return f'{self.__class__.__name__}({self.num_outputs})'
 
 
class RandomCutOut(object):
    """Cut out a random number of rectangular holes of random sizes from input image 
    at random locations.

    :param min_n_holes: minimum number of holes to be cutout from input image
    :param max_n_holes: maximum number of holes to be cutout from input image
    :param min_size: minimum size of holes to be cutout from input image
    :param max_size: maximum size of holes to be cutout from input image
    """

    def __init__(self, min_n_holes=5, max_n_holes=10,
                 min_size=5, max_size=15):
        
        self.min_n = min_n_holes
        self.max_n = max_n_holes
        self.min_size = min_size
        self.max_size = max_size
    


    def __call__(self, x):
        x = np.array(x)
        h, w, c = x.shape
        n_holes = np.random.randint(self.min_n, self.max_n)
        for n in range(n_holes):
            h_length = np.random.randint(self.min_size, self.max_size)
            w_length = np.random.randint(self.min_size, self.max_size)
            h_y = np.random.randint(0, h)
            h_x = np.random.randint(0, w)
            y1 = int(np.clip(h_y - h_length / 2, 0, h))
            y2 = int(np.clip(h_y + h_length / 2, 0, h))
            x1 = int(np.clip(h_x - w_length / 2, 0, w))
            x2 = int(np.clip(h_x + w_length / 2, 0, w))
            x[y1:y2, x1:x2, :] = 0
        return Image.fromarray(x)
    
    
    def __repr__(self):
        return f'{self.__class__.__name__}({self.min_n}, {self.max_n}, {self.min_size}, {self.min_size})'