from mpi4py import MPI
import numpy as np

from stable_baselines.common import zipsame


def mpi_mean(arr, axis=0, comm=None, keepdims=False):
    """
    calculates the mean of an array, using MPI

    :param arr: (np.ndarray)
    :param axis: (int or tuple or list) the axis to run the means over
    :param comm: (MPI Communicators) if None, MPI.COMM_WORLD
    :param keepdims: (bool) keep the other dimensions intact
    :return: (np.ndarray or Number) the result of the sum
    """
    arr = np.asarray(arr)
    assert arr.ndim > 0
    if comm is None:
        comm = MPI.COMM_WORLD
    xsum = arr.sum(axis=axis, keepdims=keepdims)
    size = xsum.size
    localsum = np.zeros(size + 1, arr.dtype)
    localsum[:size] = xsum.ravel()
    localsum[size] = arr.shape[axis]
    globalsum = np.zeros_like(localsum)
    comm.Allreduce(localsum, globalsum, op=MPI.SUM)
    return globalsum[:size].reshape(xsum.shape) / globalsum[size], globalsum[size]


def mpi_moments(arr, axis=0, comm=None, keepdims=False):
    """
    calculates the mean and std of an array, using MPI

    :param arr: (np.ndarray)
    :param axis: (int or tuple or list) the axis to run the moments over
    :param comm: (MPI Communicators) if None, MPI.COMM_WORLD
    :param keepdims: (bool) keep the other dimensions intact
    :return: (np.ndarray or Number) the result of the moments
    """
    arr = np.asarray(arr)
    assert arr.ndim > 0
    mean, count = mpi_mean(arr, axis=axis, comm=comm, keepdims=True)
    sqdiffs = np.square(arr - mean)
    meansqdiff, count1 = mpi_mean(sqdiffs, axis=axis, comm=comm, keepdims=True)
    assert count1 == count
    std = np.sqrt(meansqdiff)
    if not keepdims:
        newshape = mean.shape[:axis] + mean.shape[axis+1:]
        mean = mean.reshape(newshape)
        std = std.reshape(newshape)
    return mean, std, count


def _helper_runningmeanstd():
    comm = MPI.COMM_WORLD
    np.random.seed(0)
    for (triple, axis) in [
         ((np.random.randn(3), np.random.randn(4), np.random.randn(5)), 0),
         ((np.random.randn(3, 2), np.random.randn(4, 2), np.random.randn(5, 2)), 0),
         ((np.random.randn(2, 3), np.random.randn(2, 4), np.random.randn(2, 4)), 1)]:

        arr = np.concatenate(triple, axis=axis)
        ms1 = [arr.mean(axis=axis), arr.std(axis=axis), arr.shape[axis]]

        ms2 = mpi_moments(triple[comm.Get_rank()], axis=axis)

        for (res_1, res_2) in zipsame(ms1, ms2):
            print(res_1, res_2)
            assert np.allclose(res_1, res_2)
            print("ok!")
