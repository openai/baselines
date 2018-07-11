import numpy as np


class RunningStat(object):
    def __init__(self, shape):
        """
        calulates the running mean and std of a data stream
        http://www.johndcook.com/blog/standard_deviation/
        :param shape: (tuple) the shape of the data stream's output
        """
        self._n = 0
        self._M = np.zeros(shape)
        self._S = np.zeros(shape)

    def push(self, x):
        """
        update the running mean and std
        :param x: (numpy Number) the data
        """
        x = np.asarray(x)
        assert x.shape == self._M.shape
        self._n += 1
        if self._n == 1:
            self._M[...] = x
        else:
            old_m = self._M.copy()
            self._M[...] = old_m + (x - old_m) / self._n
            self._S[...] = self._S + (x - old_m) * (x - self._M)

    @property
    def n(self):
        """
        the number of data points
        :return: (int)
        """
        return self._n

    @property
    def mean(self):
        """
        the average value
        :return: (float)
        """
        return self._M

    @property
    def var(self):
        """
        the variation of the data points
        :return: (float)
        """
        return self._S / (self._n - 1) if self._n > 1 else np.square(self._M)

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
        return self._M.shape


def test_running_stat():
    """
    test RunningStat object
    """
    for shp in ((), (3,), (3, 4)):
        li = []
        rs = RunningStat(shp)
        for _ in range(5):
            val = np.random.randn(*shp)
            rs.push(val)
            li.append(val)
            m = np.mean(li, axis=0)
            assert np.allclose(rs.mean, m)
            v = np.square(m) if (len(li) == 1) else np.var(li, ddof=1, axis=0)
            assert np.allclose(rs.var, v)
