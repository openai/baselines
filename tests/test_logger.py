import pytest

from stable_baselines.logger import make_output_format, read_tb, read_csv, read_json, _demo


KEY_VALUES = {'test': 1, 'b': -3.14, '8': 9.9}
LOG_DIR = '/tmp/openai_baselines/'


def test_main():
    """
    Dry-run python -m stable_baselines.logger
    """
    _demo()


@pytest.mark.parametrize('_format', ['tensorboard', 'stdout', 'log', 'json', 'csv'])
def test_make_output(_format):
    """
    test make output

    :param _format: (str) output format
    """
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
