import random
import numpy as np


"""
Do not always perform a certain operation (only with a certain probability).
"""
class Sometimes:
    def __init__(self, prob, operation):
        self._prob = prob
        self._operation = operation

    def apply(self, im, random_seed=None, mask=False):
        random.seed(random_seed)

        if random.uniform(0, 1) <= self._prob:
            return self._operation.apply(im, random_seed=random.uniform(0, 1), mask=mask)
        else:
            return im


"""
Choose one of these operations. A list of operations should be provided.
"""
class ChooseOne:
    def __init__(self, operations):
        self._operations = operations

    def apply(self, im, random_seed=None, mask=False):
        random.seed(random_seed)

        chosen_operation = random.choice(self._operations)

        return chosen_operation.apply(im, random_seed=random.uniform(0, 1), mask=mask)


"""
Choose some of these operations. A minimum and maximum number of chosen operations needs to be specified.
"""
class ChooseSome:
    def __init__(self, minimum, maximum, operations):
        self._minimum = minimum
        self._maximum = maximum
        self._operations = operations

    def apply(self, im, random_seed=None, mask=False):
        random.seed(random_seed)

        np.random.seed(int(2147483647 * random_seed))

        chosen_number_of_operations = random.randint(self._minimum, self._maximum)

        chosen_operations = np.random.permutation(self._operations)[:chosen_number_of_operations]

        im_out = im

        for op in chosen_operations:
            im_out = op.apply(im_out, random_seed=random.uniform(0, 1), mask=mask)

        return im_out
