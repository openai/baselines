from .running_stat import RunningStat
from collections import deque
import numpy as np


class Filter(object):
    """
    takes a value 'x' (numpy Number), applies the filter, and returns the new value.

    Can pass kwarg: 'update' (bool) if the filter can update from the value
    """
    def __call__(self, x, update=True):
        raise NotImplementedError

    def reset(self):
        """
        resets the filter
        """
        pass

    def output_shape(self, input_space):
        """
        returns the output shape
        :param input_space: (numpy int)
        :return: (numpy int) output shape
        """
        raise NotImplementedError


class IdentityFilter(Filter):
    """
    A filter that implements an identity function

    takes a value 'x' (numpy Number), applies the filter, and returns the new value.

    Can pass kwarg: 'update' (bool) if the filter can update from the value
    """
    def __call__(self, x, update=True):
        return x

    def output_shape(self, input_space):
        return input_space.shape


class CompositionFilter(Filter):
    def __init__(self, fs):
        """
        A filter that implements a composition with other functions

        takes a value 'x' (numpy Number), applies the filter, and returns the new value.

        Can pass kwarg: 'update' (bool) if the filter can update from the value

        :param fs: ([function]) composition of these functions and the input
        """
        self.fs = fs

    def __call__(self, x, update=True):
        for f in self.fs:
            x = f(x)
        return x

    def output_shape(self, input_space):
        out = input_space.shape
        for f in self.fs:
            out = f.output_shape(out)
        return out


class ZFilter(Filter):
    def __init__(self, shape, demean=True, destd=True, clip=10.0):
        """
        A filter that implements a z-filter
        y = (x-mean)/std
        using running estimates of mean,std

        takes a value 'x' (numpy Number), applies the filter, and returns the new value.

        Can pass kwarg: 'update' (bool) if the filter can update from the value

        :param shape: ([int]) the shape of the input
        :param demean: (bool) filter mean
        :param destd: (bool) filter standard deviation
        :param clip: (float) clip filter absolute value to this value
        """
        self.demean = demean
        self.destd = destd
        self.clip = clip

        self.rs = RunningStat(shape)

    def __call__(self, x, update=True):
        if update:
            self.rs.push(x)
        if self.demean:
            x = x - self.rs.mean
        if self.destd:
            x = x / (self.rs.std + 1e-8)
        if self.clip:
            x = np.clip(x, -self.clip, self.clip)
        return x

    def output_shape(self, input_space):
        return input_space.shape


class AddClock(Filter):
    def __init__(self):
        """
        A filter that appends a counter to the input

        takes a value 'x' (numpy Number), applies the filter, and returns the new value.

        Can pass kwarg: 'update' (bool) if the filter can update from the value
        """
        self.count = 0

    def reset(self):
        self.count = 0

    def __call__(self, x, update=True):
        return np.append(x, self.count / 100.0)

    def output_shape(self, input_space):
        return input_space.shape[0] + 1,


class FlattenFilter(Filter):
    """
    A filter that flattens the input

    takes a value 'x' (numpy Number), applies the filter, and returns the new value.

    Can pass kwarg: 'update' (bool) if the filter can update from the value
    """
    def __call__(self, x, update=True):
        return x.ravel()

    def output_shape(self, input_space):
        return int(np.prod(input_space.shape)),


class Ind2OneHotFilter(Filter):
    def __init__(self, n):
        """
        A filter that turns indices to onehot encoding

        takes a value 'x' (numpy Number), applies the filter, and returns the new value.

        Can pass kwarg: 'update' (bool) if the filter can update from the value

        :param n: (int) the number of categories
        """
        self.n = n

    def __call__(self, x, update=True):
        out = np.zeros(self.n)
        out[x] = 1
        return out

    def output_shape(self, input_space):
        return input_space.n,


class DivFilter(Filter):
    def __init__(self, divisor):
        """
        A filter that divides the input from a value

        takes a value 'x' (numpy Number), applies the filter, and returns the new value.

        Can pass kwarg: 'update' (bool) if the filter can update from the value

        :param divisor: (float) the number you want to divide by
        """
        self.divisor = divisor

    def __call__(self, x, update=True):
        return x / self.divisor

    def output_shape(self, input_space):
        return input_space.shape


class StackFilter(Filter):
    def __init__(self, length):
        """
        A filter that runs a stacking of a 'length' inputs

        takes a value 'x' (numpy Number), applies the filter, and returns the new value.

        Can pass kwarg: 'update' (bool) if the filter can update from the value

        :param length: (int) the number of inputs to stack
        """
        self.stack = deque(maxlen=length)

    def reset(self):
        self.stack.clear()

    def __call__(self, x, update=True):
        self.stack.append(x)
        while len(self.stack) < self.stack.maxlen:
            self.stack.append(x)
        return np.concatenate(self.stack, axis=-1)

    def output_shape(self, input_space):
        return input_space.shape[:-1] + (input_space.shape[-1] * self.stack.maxlen,)
