from baselines.common import mpi_util
from baselines import logger
from baselines.common.tests.test_with_mpi import with_mpi
try:
    from mpi4py import MPI
except ImportError:
    MPI = None

@with_mpi()
def test_mpi_weighted_mean():
    comm = MPI.COMM_WORLD
    with logger.scoped_configure(comm=comm):
        if comm.rank == 0:
            name2valcount = {'a' : (10, 2), 'b' : (20,3)}
        elif comm.rank == 1:
            name2valcount = {'a' : (19, 1), 'c' : (42,3)}
        else:
            raise NotImplementedError
        d = mpi_util.mpi_weighted_mean(comm, name2valcount)
        correctval = {'a' : (10 * 2 + 19) / 3.0, 'b' : 20, 'c' : 42}
        if comm.rank == 0:
            assert d == correctval, '{} != {}'.format(d, correctval)

        for name, (val, count) in name2valcount.items():
            for _ in range(count):
                logger.logkv_mean(name, val)
        d2 = logger.dumpkvs()
        if comm.rank == 0:
            assert d2 == correctval
