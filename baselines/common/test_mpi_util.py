from baselines.common import mpi_util
from mpi4py import MPI
import subprocess
import sys
from baselines import logger

def helper_for_mpi_weighted_mean():
    comm = MPI.COMM_WORLD
    if comm.rank == 0:
        name2valcount = {'a' : (10, 2), 'b' : (20,3)}
    elif comm.rank == 1:
        name2valcount = {'a' : (19, 1), 'c' : (42,3)}
    else:
        raise NotImplementedError

    d = mpi_util.mpi_weighted_mean(comm, name2valcount)
    correctval = {'a' : (10 * 2 + 19) / 3.0, 'b' : 20, 'c' : 42}
    if comm.rank == 0:
        assert d == correctval, f'{d} != {correctval}'

    for name, (val, count) in name2valcount.items():
        for _ in range(count):
            logger.logkv_mean(name, val)
    d2 = logger.dumpkvs(mpi_mean=True)
    if comm.rank == 0:
        assert d2 == correctval


def test_mpi_weighted_mean():
    subprocess.check_call(['mpirun', '-n', '2', sys.executable, '-c',
        'from baselines.common import test_mpi_util; test_mpi_util.helper_for_mpi_weighted_mean()'])
