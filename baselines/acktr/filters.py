from .running_stat import RunningStat
from collections import deque
import numpy as np

class Filter(object):
    def __call__(self, x, update=True):
        raise NotImplementedError
    def reset(self):
        pass

class IdentityFilter(Filter):
    def __call__(self, x, update=True):
        return x

class CompositionFilter(Filter):
    def __init__(self, fs):
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
    """
    y = (x-mean)/std
    using running estimates of mean,std
    """

    def __init__(self, shape, demean=True, destd=True, clip=10.0):
        self.demean = demean
        self.destd = destd
        self.clip = clip

        self.rs = RunningStat(shape)

    def __call__(self, x, update=True):
        if update: self.rs.push(x)
        if self.demean:
            x = x - self.rs.mean
        if self.destd:
            x = x / (self.rs.std+1e-8)
        if self.clip:
            x = np.clip(x, -self.clip, self.clip)
        return x
    def output_shape(self, input_space):
        return input_space.shape

class AddClock(Filter):
    def __init__(self):
        self.count = 0
    def reset(self):
        self.count = 0
    def __call__(self, x, update=True):
        return np.append(x, self.count/100.0)
    def output_shape(self, input_space):
        return (input_space.shape[0]+1,)

class FlattenFilter(Filter):
    def __call__(self, x, update=True):
        return x.ravel()
    def output_shape(self, input_space):
        return (int(np.prod(input_space.shape)),)

class Ind2OneHotFilter(Filter):
    def __init__(self, n):
        self.n = n
    def __call__(self, x, update=True):
        out = np.zeros(self.n)
        out[x] = 1
        return out
    def output_shape(self, input_space):
        return (input_space.n,)

class DivFilter(Filter):
    def __init__(self, divisor):
        self.divisor = divisor
    def __call__(self, x, update=True):
        return x / self.divisor
    def output_shape(self, input_space):
        return input_space.shape

class StackFilter(Filter):
    def __init__(self, length):
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
