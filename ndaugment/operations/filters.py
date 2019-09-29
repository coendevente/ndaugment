from random import seed
from ndaugment.utils.utils import *
import numpy as np
from scipy.ndimage.filters import gaussian_filter


class GaussianBlur:
    def __init__(self, sigma, **opt_params):
        self._sigma = sigma
        self._opt_params = opt_params

    def apply(self, im, random_seed=None, mask=False):
        seed(random_seed)

        sigma = convert_tuple_of_tuples(self._sigma, im.ndim)
        if mask:
            return im

        return gaussian_filter(im, sigma, **self._opt_params)


def normalize(im):
    if np.max(im) - np.min(im) == 0:
        return im
    return (im - np.min(im)) / (np.max(im) - np.min(im))
