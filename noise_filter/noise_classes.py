from skimage import util

import numpy as np
from PIL import Image

def make_noisy_gaussian(images, mean, var, seed=None):
    noisy_images =  util.random_noise(images, seed=seed, mode='gaussian', mean=mean, var=var)
    return np.uint8(noisy_images*255)

class AddGaussianNoise:
    def __init__(self, mean=0, var=1, seed=None) -> None:
        self.mean = mean
        self.var = var
        self.seed = seed

    def __call__(self, pilimage):
        tensor = np.asarray(pilimage)
        np_image = make_noisy_gaussian(images=tensor, mean=self.mean, var=self.var, seed=self.seed)
        return Image.fromarray(np_image)

    def __repr__(self) -> str:
        return f'Gaussian noise: mean={self.mean}, var={self.var}'