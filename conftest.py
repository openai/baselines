"""Configures pytest to ignore certain unit tests unless the appropriate flag is passed.

--rungpu: tests that require GPU.
--expensive: tests that take a long time to run (e.g. training an RL algorithm for many timestesps)."""

import pytest


def pytest_addoption(parser):
    parser.addoption("--rungpu", action="store_true", default=False, help="run gpu tests")
    parser.addoption("--expensive", action="store_true",
                     help="run expensive tests (which are otherwise skipped).")


def pytest_collection_modifyitems(config, items):
    flags = {'gpu': '--rungpu', 'expensive': '--expensive'}
    skips = {keyword: pytest.mark.skip(reason="need {} option to run".format(flag))
             for keyword, flag in flags.items() if not config.getoption(flag)}
    for item in items:
        for keyword, skip in skips.items():
            if keyword in item.keywords:
                item.add_marker(skip)
