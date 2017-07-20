import os, subprocess, sys

def mpi_fork(n):
    """Re-launches the current script with workers
    Returns "parent" for original parent, "child" for MPI children
    """
    if n<=1: 
        return "child"
    if os.getenv("IN_MPI") is None:
        env = os.environ.copy()
        env.update(
            MKL_NUM_THREADS="1",
            OMP_NUM_THREADS="1",
            IN_MPI="1"
        )
        subprocess.check_call(["mpirun", "-np", str(n), sys.executable] + sys.argv, env=env)
        return "parent"
    else:
        return "child"
