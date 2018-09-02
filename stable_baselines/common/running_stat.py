import numpy as np


class RunningStat(object):
    def __init__(self, shape):
        """
        calulates the running mean and std of a data stream
        http://www.johndcook.com/blog/standard_deviation/

        :param shape: (tuple) the shape of the data stream's output
        """
        self._step = 0
        self._mean = np.zeros(shape)
        self._std = np.zeros(shape)

    def push(self, value):
        """
        update the running mean and std

        :param value: (np.ndarray) the data
        """
        value = np.asarray(value)
        assert value.shape == self._mean.shape
        self._step += 1
        if self._step == 1:
            self._mean[...] = value
        else:
            old_m = self._mean.copy()
            self._mean[...] = old_m + (value - old_m) / self._step
            self._std[...] = self._std + (value - old_m) * (value - self._mean)

    @property
    def n(self):
        """
        the number of data points

        :return: (int)
        """
        return self._step

    @property
    def mean(self):
        """
        the average value

        :return: (float)
        """
        return self._mean

    @property
    def var(self):
        """
        the variation of the data points

        :return: (float)
        """
        return self._std / (self._step - 1) if self._step > 1 else np.square(self._mean)

    @property
    def std(self):
        """
        the standard deviation of the data points

        :return: (float)
        """
        return np.sqrt(self.var)

    @property
    def shape(self):
        """
        the shape of the data points

        :return: (tuple)
        """
        return self._mean.shape
