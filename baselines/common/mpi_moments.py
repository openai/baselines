from mpi4py import MPI
import numpy as np
from baselines.common import zipsame

def mpi_moments(x, axis=0):
    x = np.asarray(x, dtype='float64')
    newshape = list(x.shape)
    newshape.pop(axis)
    n = np.prod(newshape,dtype=int)
    totalvec = np.zeros(n*2+1, 'float64')
    addvec = np.concatenate([x.sum(axis=axis).ravel(), 
        np.square(x).sum(axis=axis).ravel(), 
        np.array([x.shape[axis]],dtype='float64')])
    MPI.COMM_WORLD.Allreduce(addvec, totalvec, op=MPI.SUM)
    sum = totalvec[:n]
    sumsq = totalvec[n:2*n]
    count = totalvec[2*n]
    if count == 0:
        mean = np.empty(newshape); mean[:] = np.nan
        std = np.empty(newshape); std[:] = np.nan
    else:
        mean = sum/count
        std = np.sqrt(np.maximum(sumsq/count - np.square(mean),0))
    return mean, std, count


def test_runningmeanstd():
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

if __name__ == "__main__":
    #mpirun -np 3 python <script>
    test_runningmeanstd()