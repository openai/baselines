import pytest


def pytest_addoption(parser):
    parser.addoption('--runslow', action='store_true', default=False, help='run slow tests')


def pytest_collection_modifyitems(config, items):
    if config.getoption('--runslow'):
        # --runslow given in cli: do not skip slow tests
        return
    skip_slow = pytest.mark.skip(reason='need --runslow option to run')
    slow_tests = []
    for item in items:
        if 'slow' in item.keywords:
            slow_tests.append(item.name)
            item.add_marker(skip_slow)

    print('skipping slow tests', ' '.join(slow_tests), 'use --runslow to run this')
