import copy
import sys
import random


class Sequential:
    def __init__(self, sequence=False):
        self._sequence = sequence if sequence else []

    def add(self, operation):
        self._sequence.append(operation)

    def get_sequence(self):
        return self._sequence

    def apply(self, im, random_seed=None, mask=False):
        random.seed(random_seed)
        im_out = copy.copy(im)
        for operation in self._sequence:
            im_out = operation.apply(im_out, random_seed=random.uniform(0, 1), mask=mask)
        return im_out

    # verbose 0 (no output) or 1 (output)
    def apply_multiple(self, imgs, random_seed=None, mask=False, freq=1, verbose=0):
        random.seed(random_seed)
        out = []
        p = ProgressBar(freq * len(imgs))
        if verbose == 1:
            p.setup()
        for i in range(freq):
            for j, im in enumerate(imgs):
                out.append(self.apply(im, random_seed=random.uniform(0, 1), mask=mask))
                if verbose == 1:
                    p.update(i * len(imgs) + j + 1)
        if verbose == 1:
            p.finish()
        return out

    def __add__(self, other):
        return Sequential(self._sequence + other.get_sequence())


class ProgressBar:
    def __init__(self, n_iter, width=40):
        self._n_iter = n_iter
        self._width = width
        self._last = 0

    def setup(self):
        # setup toolbar
        sys.stdout.write('%s' % (' ' * self._width))
        sys.stdout.flush()
        sys.stdout.write('\b' * (self._width + 1))  # return to start of line, after '['

    def update(self, n_current):
        current = round(n_current / self._n_iter * self._width)
        if current > self._last:
            p = '█' * (current - self._last)
            sys.stdout.write(p)
            sys.stdout.flush()
            self._last = current

    @staticmethod
    def finish():
        sys.stdout.write('\n')
