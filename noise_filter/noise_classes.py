from noise_filter import make_noisy_blurry as mn
import numpy as np
from PIL import Image

class AddGaussianNoise:
    def __init__(self, mean=0, var=1, seed=None) -> None:
        self.mean = mean
        self.var = var
        self.seed = seed

    def __call__(self, pilimage):
        tensor = np.asarray(pilimage)
        np_image = mn.make_noisy_gaussian(images=tensor, mean=self.mean, var=self.var, seed=self.seed)
        return Image.fromarray(np_image)

    def __repr__(self) -> str:
        return f'Gaussian noise: mean={self.mean}, var={self.var}'