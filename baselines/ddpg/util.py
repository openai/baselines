import numpy as np
import tensorflow as tf
from mpi4py import MPI
from baselines.common.mpi_moments import mpi_moments


def reduce_var(x, axis=None, keepdims=False):
    m = tf.reduce_mean(x, axis=axis, keep_dims=True)
    devs_squared = tf.square(x - m)
    return tf.reduce_mean(devs_squared, axis=axis, keep_dims=keepdims)


def reduce_std(x, axis=None, keepdims=False):
    return tf.sqrt(reduce_var(x, axis=axis, keepdims=keepdims))


def mpi_mean(value):
    if value == []:
        value = [0.]
    if not isinstance(value, list):
        value = [value]
    return mpi_moments(np.array(value))[0][0]


def mpi_std(value):
    if value == []:
        value = [0.]
    if not isinstance(value, list):
        value = [value]
    return mpi_moments(np.array(value))[1][0]


def mpi_max(value):
    global_max = np.zeros(1, dtype='float64')
    local_max = np.max(value).astype('float64')
    MPI.COMM_WORLD.Reduce(local_max, global_max, op=MPI.MAX)
    return global_max[0]


def mpi_sum(value):
    global_sum = np.zeros(1, dtype='float64')
    local_sum = np.sum(np.array(value)).astype('float64')
    MPI.COMM_WORLD.Reduce(local_sum, global_sum, op=MPI.SUM)
    return global_sum[0]
