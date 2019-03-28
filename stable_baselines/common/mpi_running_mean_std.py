from mpi4py import MPI
import tensorflow as tf
import numpy as np

import stable_baselines.common.tf_util as tf_util


class RunningMeanStd(object):
    def __init__(self, epsilon=1e-2, shape=()):
        """
        calulates the running mean and std of a data stream
        https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm

        :param epsilon: (float) helps with arithmetic issues
        :param shape: (tuple) the shape of the data stream's output
        """
        self._sum = tf.get_variable(
            dtype=tf.float64,
            shape=shape,
            initializer=tf.constant_initializer(0.0),
            name="runningsum", trainable=False)
        self._sumsq = tf.get_variable(
            dtype=tf.float64,
            shape=shape,
            initializer=tf.constant_initializer(epsilon),
            name="runningsumsq", trainable=False)
        self._count = tf.get_variable(
            dtype=tf.float64,
            shape=(),
            initializer=tf.constant_initializer(epsilon),
            name="count", trainable=False)
        self.shape = shape

        self.mean = tf.cast(self._sum / self._count, tf.float32)
        self.std = tf.sqrt(tf.maximum(tf.cast(self._sumsq / self._count, tf.float32) - tf.square(self.mean),
                                      1e-2))

        newsum = tf.placeholder(shape=self.shape, dtype=tf.float64, name='sum')
        newsumsq = tf.placeholder(shape=self.shape, dtype=tf.float64, name='var')
        newcount = tf.placeholder(shape=[], dtype=tf.float64, name='count')
        self.incfiltparams = tf_util.function([newsum, newsumsq, newcount], [],
                                              updates=[tf.assign_add(self._sum, newsum),
                                                       tf.assign_add(self._sumsq, newsumsq),
                                                       tf.assign_add(self._count, newcount)])

    def update(self, data):
        """
        update the running mean and std

        :param data: (np.ndarray) the data
        """
        data = data.astype('float64')
        data_size = int(np.prod(self.shape))
        totalvec = np.zeros(data_size * 2 + 1, 'float64')
        addvec = np.concatenate([data.sum(axis=0).ravel(), np.square(data).sum(axis=0).ravel(),
                                 np.array([len(data)], dtype='float64')])
        MPI.COMM_WORLD.Allreduce(addvec, totalvec, op=MPI.SUM)
        self.incfiltparams(totalvec[0: data_size].reshape(self.shape),
                           totalvec[data_size: 2 * data_size].reshape(self.shape), totalvec[2 * data_size])


@tf_util.in_session
def test_dist():
    """
    test the running mean std
    """
    np.random.seed(0)
    p_1, p_2, p_3 = (np.random.randn(3, 1), np.random.randn(4, 1), np.random.randn(5, 1))
    q_1, q_2, q_3 = (np.random.randn(6, 1), np.random.randn(7, 1), np.random.randn(8, 1))

    comm = MPI.COMM_WORLD
    assert comm.Get_size() == 2
    if comm.Get_rank() == 0:
        x_1, x_2, x_3 = p_1, p_2, p_3
    elif comm.Get_rank() == 1:
        x_1, x_2, x_3 = q_1, q_2, q_3
    else:
        assert False

    rms = RunningMeanStd(epsilon=0.0, shape=(1,))
    tf_util.initialize()

    rms.update(x_1)
    rms.update(x_2)
    rms.update(x_3)

    bigvec = np.concatenate([p_1, p_2, p_3, q_1, q_2, q_3])

    def checkallclose(var_1, var_2):
        print(var_1, var_2)
        return np.allclose(var_1, var_2)

    assert checkallclose(
        bigvec.mean(axis=0),
        rms.mean.eval(),
    )
    assert checkallclose(
        bigvec.std(axis=0),
        rms.std.eval(),
    )


if __name__ == "__main__":
    # Run with mpirun -np 2 python <filename>
    test_dist()
