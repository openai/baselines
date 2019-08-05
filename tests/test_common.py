from contextlib import contextmanager
import sys


def _assert_eq(left, right):
    assert left == right, '{} != {}'.format(left, right)


def _assert_neq(left, right):
    assert left != right, '{} == {}'.format(left, right)


@contextmanager
def _maybe_disable_mpi(mpi_disabled):
    """A context that can temporarily remove the mpi4py import.

    Useful for testing whether non-MPI algorithms work as intended when
    mpi4py isn't installed.

    Args:
        disable_mpi (bool): If True, then this context temporarily removes
            the mpi4py import from `sys.modules`
    """
    if mpi_disabled and "mpi4py" in sys.modules:
        temp = sys.modules["mpi4py"]
        try:
            sys.modules["mpi4py"] = None
            yield
        finally:
            sys.modules["mpi4py"] = temp
    else:
        yield
