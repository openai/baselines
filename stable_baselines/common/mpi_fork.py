import os
import subprocess
import sys


def mpi_fork(rank, bind_to_core=False):
    """
    Re-launches the current script with workers
    Returns "parent" for original parent, "child" for MPI children

    :param rank: (int) the rank
    :param bind_to_core: (bool) enables binding to core
    :return: (str) the correct type of thread name
    """
    if rank <= 1:
        return "child"
    if os.getenv("IN_MPI") is None:
        env = os.environ.copy()
        env.update(
            MKL_NUM_THREADS="1",
            OMP_NUM_THREADS="1",
            IN_MPI="1"
        )
        args = ["mpirun", "-np", str(rank)]
        if bind_to_core:
            args += ["-bind-to", "core"]
        args += [sys.executable] + sys.argv
        subprocess.check_call(args, env=env)
        return "parent"
    else:
        return "child"
