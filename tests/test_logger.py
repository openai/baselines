import pytest
import numpy as np

from stable_baselines.logger import make_output_format, read_tb, read_csv, read_json, _demo
from .test_common import _maybe_disable_mpi


KEY_VALUES = {
    "test": 1,
    "b": -3.14,
    "8": 9.9,
    "l": [1, 2],
    "a": np.array([1, 2, 3]),
    "f": np.array(1),
    "g": np.array([[[1]]]),
}
LOG_DIR = '/tmp/openai_baselines/'


def test_main():
    """
    Dry-run python -m stable_baselines.logger
    """
    _demo()


@pytest.mark.parametrize('_format', ['tensorboard', 'stdout', 'log', 'json', 'csv'])
@pytest.mark.parametrize('mpi_disabled', [False, True])
def test_make_output(_format, mpi_disabled):
    """
    test make output

    :param _format: (str) output format
    """
    with _maybe_disable_mpi(mpi_disabled):
        writer = make_output_format(_format, LOG_DIR)
        writer.writekvs(KEY_VALUES)
        if _format == 'tensorboard':
            read_tb(LOG_DIR)
        elif _format == "csv":
            read_csv(LOG_DIR + 'progress.csv')
        elif _format == 'json':
            read_json(LOG_DIR + 'progress.json')
        writer.close()


def test_make_output_fail():
    """
    test value error on logger
    """
    with pytest.raises(ValueError):
        make_output_format('dummy_format', LOG_DIR)
