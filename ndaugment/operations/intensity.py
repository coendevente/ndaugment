from random import seed
from random import uniform as r
import numpy as np
import copy


class ChangeContrast:
    def __init__(self, power):
        self._power = power

    def apply(self, im, random_seed=None, mask=False):
        seed(random_seed)
        power = r(self._power[0], self._power[1])

        if mask:
            return im

        a, b = .5, 1.5

        out = copy.copy(im)
        amplitude_pre = np.max(out) - np.min(out)
        min_pre = np.min(out)

        out = normalize(out) * (b - a) + a
        out = np.power(out, power)
        out = normalize(out) * amplitude_pre + min_pre

        return out


def normalize(im):
    if np.max(im) - np.min(im) == 0:
        return im
    return (im - np.min(im)) / (np.max(im) - np.min(im))
