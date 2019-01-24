import os
import sys
import subprocess
import cloudpickle
import base64
import pytest

try:
    from mpi4py import MPI
except ImportError:
    MPI = None

def with_mpi(nproc=2, timeout=30, skip_if_no_mpi=True):
    def outer_thunk(fn):
        def thunk(*args, **kwargs):
            serialized_fn = base64.b64encode(cloudpickle.dumps(lambda: fn(*args, **kwargs)))
            subprocess.check_call([
                'mpiexec','-n', str(nproc),
                sys.executable,
                '-m', 'baselines.common.tests.test_with_mpi',
                serialized_fn
            ], env=os.environ, timeout=timeout)

        if skip_if_no_mpi:
            return pytest.mark.skipif(MPI is None, reason="MPI not present")(thunk)
        else:
            return thunk

    return outer_thunk


if __name__ == '__main__':
    if len(sys.argv) > 1:
        fn = cloudpickle.loads(base64.b64decode(sys.argv[1]))
        assert callable(fn)
        fn()
