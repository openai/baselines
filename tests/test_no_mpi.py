import sys

from .test_common import _maybe_disable_mpi


def test_no_mpi_no_crash():
    with _maybe_disable_mpi(True):
        # Temporarily delete previously imported stable baselines
        old_modules = {}
        sb_modules = [name for name in sys.modules.keys()
                      if name.startswith('stable_baselines')]
        for name in sb_modules:
            old_modules[name] = sys.modules.pop(name)

        # Re-import (with mpi disabled)
        import stable_baselines
        del stable_baselines  # appease Codacy

        # Restore old version of stable baselines (with MPI imported)
        for name, mod in old_modules.items():
            sys.modules[name] = mod
