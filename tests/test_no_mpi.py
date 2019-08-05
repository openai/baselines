from .test_common import _maybe_disable_mpi

def test_no_mpi_no_crash():
    with _maybe_disable_mpi(True):
        import stable_baselines
        del stable_baselines  # keep Codacy happy
