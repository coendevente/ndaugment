import random
import collections
import math
import numpy as np
import copy


def crop_pad_around_center(im, sh, opt_params):
    n_dim = len(sh)
    out = copy.copy(im)

    padder = [(0, 0)] * n_dim  # For padding
    for i, o_s, i_s in zip(range(n_dim), im.shape, sh):
        diff = o_s - i_s
        if diff == 0:
            continue
        l = int(math.floor(abs(diff) / 2))
        r = abs(diff) - l
        if diff < 0:
            padder[i] = (l, r)
        else:  # diff < 0
            out = out.swapaxes(0, i)[l:o_s - r].swapaxes(i, 0)

    op = change_params(opt_params)
    out = np.pad(out, padder, **op)

    return out


def change_params(opt_params):
    op = opt_params
    if 'mode' not in op.keys():
        op['mode'] = 'constant'
    if 'cval' in op:
        op['constant_values'] = op['cval']
        del op['cval']

    return op


"""
Will choose random values in specified range. 
"""
def convert_tuple_of_tuples(t, n_dim):
    if isinstance(t[0], collections.Iterable):
        assert len(t) == n_dim, 'len(self._val) != len(im.shape) ({} != {})'. \
            format(len(t), n_dim)
        return [random.uniform(v[0], v[1]) for v in t]
    else:
        return [random.uniform(t[0], t[1])] * n_dim