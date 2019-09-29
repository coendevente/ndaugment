from scipy.ndimage import rotate, zoom, affine_transform, shift
from ndaugment.utils.utils import *
import numpy as np
import math
import copy
import random
from random import seed
from random import uniform as r
import collections
import warnings
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter

warnings.filterwarnings('ignore', '.*output shape of zoom.*')


class Rotate:
    def __init__(self, angle, *params, **opt_params):
        self._angle = angle
        self._params = params
        self._opt_params = opt_params

    def apply(self, im, random_seed=None, mask=False):
        seed(random_seed)

        angle = r(self._angle[0], self._angle[1])
        return rotate(im, angle, reshape=False, *self._params, **self._opt_params)


class Zoom:
    def __init__(self, val, *params, **opt_params):
        self._val = val  # 0.8 means 20% zooming out, 1.0 means no difference, and 1.2 means 20% zooming in
        self._params = params
        self._opt_params = opt_params

    def apply(self, im, random_seed=None, mask=False):
        seed(random_seed)

        val = convert_tuple_of_tuples(self._val, im.ndim)

        out = zoom(im, val)  # Perform the zooming
        out = crop_pad_around_center(out, im.shape, self._opt_params)
        return out


class Shear:
    def __init__(self, shear_rate, axes=(0, 1), *params, **opt_params):
        self._shear_rate = shear_rate
        self._axes = axes
        self._params = params
        self._opt_params = opt_params

    def apply(self, im, random_seed=None, mask=False):
        seed(random_seed)

        angle = r(self._shear_rate[0], self._shear_rate[1])

        val = angle

        # Unit matrix
        matrix = np.eye(len(im.shape) + 1)

        # Shift center of image to origin
        shift = np.array(im.shape) / 2
        shift_matrix = np.eye(len(im.shape) + 1).astype(np.float)
        shift_matrix[:-1, -1] = shift

        matrix = np.matmul(matrix, shift_matrix)

        # Perform shearning
        shear_matrix = np.eye(len(im.shape) + 1)
        shear_matrix[self._axes[1], self._axes[0]] = -val

        matrix = np.matmul(matrix, shear_matrix)

        # Bring center of image back to the original center (away from origin)
        shift_matrix_rev = np.eye(len(im.shape) + 1)
        shift_matrix_rev[:-1, -1] = -shift

        matrix = np.matmul(matrix, shift_matrix_rev)

        out = im
        out = affine_transform(out, matrix, **self._opt_params)
        return out


# Function to distort image
class ElasticTransform:
    def __init__(self, alpha, sigma, **opt_params):
        self._alpha = alpha
        self._sigma = sigma
        self._opt_params = opt_params

    def apply(self, im, random_seed=None, mask=False):
        """Elastic deformation of images as described in [Simard2003]_ (with modifications).
        .. [Simard2003] Simard, Steinkraus and Platt, "Best Practices for
             Convolutional Neural Networks applied to Visual Document Analysis", in
             Proc. of the International Conference on Document Analysis and
             Recognition, 2003.

         Based on https://gist.github.com/erniejunior/601cdf56d2b424757de5
         and https://www.kaggle.com/bguberfain/elastic-transform-for-data-augmentation
        """
        seed(random_seed)

        n_dim = im.ndim

        alpha = convert_tuple_of_tuples(self._alpha, n_dim)
        sigma = convert_tuple_of_tuples(self._sigma, n_dim)

        np.random.seed(int(2147483647 * random_seed))

        shape = im.shape

        # print(alpha)
        # print(sigma)

        d = list(gaussian_filter((np.random.rand(*shape) * 2 - 1), s) * a for a, s in zip(alpha, sigma))

        ran = list(range(n_dim))
        ran[0], ran[1] = ran[1], ran[0]
        x = np.meshgrid(*[np.arange(shape[i]) for i in ran])
        x[0], x[1] = x[1], x[0]
        indices = [np.reshape(xi + di, (-1, 1)) for di, xi in zip(d, x)]

        op = change_params(self._opt_params)
        out = map_coordinates(im, indices, order=1, **op).reshape(shape)
        return out


class Flip:
    def __init__(self, prob, axis):
        self._prob = prob
        self._axis = axis

    def apply(self, im, random_seed=None, mask=False):
        seed(random_seed)

        if r(0, 1) <= self._prob:
            return np.flip(im, axis=self._axis)
        else:
            return im


class Crop:
    def __init__(self, crop_size):
        self._crop_size = crop_size

    def apply(self, im, random_seed=None, mask=False):
        seed(random_seed)

        top_left_min = [0] * len(im.shape)
        top_left_max = np.array(im.shape) - np.array(self._crop_size)

        top_left = [random.randint(mn, mx) for mn, mx in zip(top_left_min, top_left_max)]
        bottom_right = np.array(top_left) + np.array(self._crop_size)

        slices = tuple(slice(a, b) for a, b in zip(top_left, bottom_right))

        cropped = im[slices]

        return cropped


class Translate:
    def __init__(self, translation_size, *params, **opt_params):
        self._translation_size = translation_size
        self._params = params
        self._opt_params = opt_params

    def apply(self, im, random_seed=None, mask=False):
        seed(random_seed)

        translation_size = convert_tuple_of_tuples(self._translation_size, im.ndim)

        shifted = shift(im, translation_size, *self._params, **self._opt_params)

        return shifted
