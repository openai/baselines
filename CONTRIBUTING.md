## Contributing to Stable-Baselines

If you are interested in contributing to Stable-Baselines, your contributions will fall
into two categories:
1. You want to propose a new Feature and implement it
    - Create an issue about your intended feature, and we shall discuss the design and
    implementation. Once we agree that the plan looks good, go ahead and implement it.
2. You want to implement a feature or bug-fix for an outstanding issue
    - Look at the outstanding issues here: https://github.com/hill-a/stable-baselines/issues
    - Look at the roadmap here: https://github.com/hill-a/stable-baselines/projects/1
    - Pick an issue or feature and comment on the task that you want to work on this feature.
    - If you need more context on a particular issue, please ask and we shall provide.

Once you finish implementing a feature or bug-fix, please send a Pull Request to
https://github.com/hill-a/stable-baselines/


If you are not familiar with creating a Pull Request, here are some guides:
- http://stackoverflow.com/questions/14680711/how-to-do-a-github-pull-request
- https://help.github.com/articles/creating-a-pull-request/


## Developing Stable-Baselines

To develop Stable-Baselines on your machine, here are some tips:

1. Clone a copy of Stable-Baselines from source:

```bash
git clone https://github.com/hill-a/stable-baselines/
cd stable-baselines
```

2. Install Stable-Baselines in develop mode, with support for building the docs and running tests:

```bash
pip install -e .[docs,tests]
```

## Codestyle

We follow the [PEP8 codestyle](https://www.python.org/dev/peps/pep-0008/). Please order the imports as follows:

1. built-in
2. packages
3. current module

with one space between each,  that gives for instance:
```python
import os
import warnings

import numpy as np

from stable_baselines import PPO2
```

In general, we recommend using pycharm to format everything in an efficient way.

Please documentation each function/method using the following template:

```python

def my_function(arg1, arg2):
    """
    Short description of the function.

    :param arg1: (arg1 type) describe what is arg1
    :param arg2: (arg2 type) describe what is arg2
    :return: (return type) describe what is returned
    """
    ...
    return my_variable
```

## Pull Request (PR)

Before proposing a PR, please open an issue, where the feature will be discussed. This prevent from duplicated PR to be proposed and also ease the code review process.

Each PR need to be reviewed and accepted by at least one of the maintainers (@hill-a , @araffin or @erniejunior ).
A PR must pass the Continuous Integration tests (travis + codacy) to be merged with the master branch.

Note: in rare cases, we can create exception for codacy failure.

## Test

All new features must add tests in the `tests/` folder ensuring that everything works fine.
We use [pytest](https://pytest.org/).
Also, when a bug fix is proposed, tests should be added to avoid regression.

To run tests:

```
./run_tests.sh
```

## Changelog and Documentation

Please do not forget to update the changelog and add documentation if needed.
A README is present in the `docs/` folder for instructions on how to build the documentation.


Credits: this contributing guide is based on the [PyTorch](https://github.com/pytorch/pytorch/) one.
