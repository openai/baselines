from mpi4py import MPI
import numpy as np
from baselines.common import zipsame


def mpi_mean(x, axis=0, comm=None, keepdims=False):
    x = np.asarray(x)
    assert x.ndim > 0
    if comm is None: comm = MPI.COMM_WORLD
    xsum = x.sum(axis=axis, keepdims=keepdims)
    n = xsum.size
    localsum = np.zeros(n+1, x.dtype)
    localsum[:n] = xsum.ravel()
    localsum[n] = x.shape[axis]
    globalsum = np.zeros_like(localsum)
    comm.Allreduce(localsum, globalsum, op=MPI.SUM)
    return globalsum[:n].reshape(xsum.shape) / globalsum[n], globalsum[n]

def mpi_moments(x, axis=0, comm=None, keepdims=False):
    x = np.asarray(x)
    assert x.ndim > 0
    mean, count = mpi_mean(x, axis=axis, comm=comm, keepdims=True)
    sqdiffs = np.square(x - mean)
    meansqdiff, count1 = mpi_mean(sqdiffs, axis=axis, comm=comm, keepdims=True)
    assert count1 == count
    std = np.sqrt(meansqdiff)
    if not keepdims:
        newshape = mean.shape[:axis] + mean.shape[axis+1:]
        mean = mean.reshape(newshape)
        std = std.reshape(newshape)
    return mean, std, count


def test_runningmeanstd():
    import subprocess
    subprocess.check_call(['mpirun', '-np', '3',
        'python','-c',
        'from baselines.common.mpi_moments import _helper_runningmeanstd; _helper_runningmeanstd()'])

def _helper_runningmeanstd():
    comm = MPI.COMM_WORLD
    np.random.seed(0)
    for (triple,axis) in [
        ((np.random.randn(3), np.random.randn(4), np.random.randn(5)),0),
        ((np.random.randn(3,2), np.random.randn(4,2), np.random.randn(5,2)),0),
        ((np.random.randn(2,3), np.random.randn(2,4), np.random.randn(2,4)),1),
        ]:


        x = np.concatenate(triple, axis=axis)
        ms1 = [x.mean(axis=axis), x.std(axis=axis), x.shape[axis]]


        ms2 = mpi_moments(triple[comm.Get_rank()],axis=axis)

        for (a1,a2) in zipsame(ms1, ms2):
            print(a1, a2)
            assert np.allclose(a1, a2)
            print("ok!")

