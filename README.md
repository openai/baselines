<img src="docs/\_static/img/logo.png" align="right" width="40%"/>

[![Build Status](https://travis-ci.com/hill-a/stable-baselines.svg?branch=master)](https://travis-ci.com/hill-a/stable-baselines) [![Documentation Status](https://readthedocs.org/projects/stable-baselines/badge/?version=master)](https://stable-baselines.readthedocs.io/en/master/?badge=master) [![Codacy Badge](https://api.codacy.com/project/badge/Grade/3bcb4cd6d76a4270acb16b5fe6dd9efa)](https://www.codacy.com/app/baselines_janitors/stable-baselines?utm_source=github.com&amp;utm_medium=referral&amp;utm_content=hill-a/stable-baselines&amp;utm_campaign=Badge_Grade) [![Codacy Badge](https://api.codacy.com/project/badge/Coverage/3bcb4cd6d76a4270acb16b5fe6dd9efa)](https://www.codacy.com/app/baselines_janitors/stable-baselines?utm_source=github.com&utm_medium=referral&utm_content=hill-a/stable-baselines&utm_campaign=Badge_Coverage)

# Stable Baselines

Stable Baselines is a set of improved implementations of reinforcement learning algorithms based on OpenAI [Baselines](https://github.com/openai/baselines/).

You can read a detailed presentation of Stable Baselines in the [Medium article](https://medium.com/@araffin/stable-baselines-a-fork-of-openai-baselines-reinforcement-learning-made-easy-df87c4b2fc82).


These algorithms will make it easier for the research community and industry to replicate, refine, and identify new ideas, and will create good baselines to build projects on top of. We expect these tools will be used as a base around which new ideas can be added, and as a tool for comparing a new approach against existing ones. We also hope that the simplicity of these tools will allow beginners to experiment with a more advanced toolset, without being buried in implementation details.

## Main differences with OpenAI Baselines

This toolset is a fork of OpenAI Baselines, with a major structural refactoring, and code cleanups:
- Unified structure for all algorithms
- PEP8 compliant (unified code style)
- Documented functions and classes
- More tests & more code coverage

| **Features**                | **Stable-Baselines**              | **OpenAI Baselines**              |
| --------------------------- | --------------------------------- | --------------------------------- |
| State of the art RL methods | :heavy_check_mark: <sup>(1)</sup> | :heavy_check_mark:                |
| Documentation               | :heavy_check_mark:                | :x:                               |
| Custom environments         | :heavy_check_mark:                | :heavy_check_mark:                |
| Custom policies             | :heavy_check_mark:                | :heavy_minus_sign: <sup>(2)</sup> |
| Common interface            | :heavy_check_mark:                | :heavy_minus_sign: <sup>(3)</sup> |
| Tensorboard support         | :heavy_check_mark:                | :heavy_minus_sign: <sup>(4)</sup> |
| Ipython / Notebook friendly | :heavy_check_mark:                | :x:                               |
| PEP8 code style             | :heavy_check_mark:                | :heavy_minus_sign: <sup>(5)</sup> |
| Custom callback             | :heavy_check_mark:                | :heavy_minus_sign: <sup>(6)</sup> |

<sup><sup>(1): Forked from previous version of OpenAI baselines, however missing refactoring for HER.</sup></sup><br>
<sup><sup>(2): Currently not available for DDPG, and only from the run script. </sup></sup><br>
<sup><sup>(3): Only via the run script.</sup></sup><br>
<sup><sup>(4): Rudimentary logging of training information (no loss nor graph). </sup></sup><br>
<sup><sup>(5): WIP on OpenAI's side (you can do it OpenAI! :cat:)</sup></sup><br>
<sup><sup>(6): Passing a callback function is only available for DQN</sup></sup><br>

## Documentation

Documentation is available online: [https://stable-baselines.readthedocs.io/](https://stable-baselines.readthedocs.io/)

## Installation

### Prerequisites
Baselines requires python3 (>=3.5) with the development headers. You'll also need system packages CMake, OpenMPI and zlib. Those can be installed as follows

#### Ubuntu

```bash
sudo apt-get update && sudo apt-get install cmake libopenmpi-dev python3-dev zlib1g-dev
```

#### Mac OS X
Installation of system packages on Mac requires [Homebrew](https://brew.sh). With Homebrew installed, run the follwing:
```bash
brew install cmake openmpi
```

### Install using pip
Install the Stable Baselines package

Using pip from pypi:
```
pip install stable-baselines
```

Please read the [documentation](https://stable-baselines.readthedocs.io/) for more details and alternatives (from source, using docker).


## Example

Most of the library tries to follow a sklearn-like syntax for the Reinforcement Learning algorithms.

Here is a quick example of how to train and run PPO2 on a cartpole environment:
```python
import gym

from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import PPO2

env = gym.make('CartPole-v1')
env = DummyVecEnv([lambda: env])  # The algorithms require a vectorized environment to run

model = PPO2(MlpPolicy, env, verbose=1)
model.learn(total_timesteps=10000)

obs = env.reset()
for i in range(1000):
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    env.render()
```

Or just train a model with a one liner if [the environment is registered in Gym](https://github.com/openai/gym/wiki/Environments) and if [the policy is registered](https://stable-baselines.readthedocs.io/en/master/guide/custom_policy.html):

```python
from stable_baselines import PPO2

model = PPO2('MlpPolicy', 'CartPole-v1').learn(10000)
```

Please read the [documentation](https://stable-baselines.readthedocs.io/) for more examples.


## Try it online with Colab Notebooks !

All the following examples can be executed online using Google colab notebooks:

- [Getting Started](https://colab.research.google.com/drive/1_1H5bjWKYBVKbbs-Kj83dsfuZieDNcFU)
- [Training, Saving, Loading](https://colab.research.google.com/drive/1KoAQ1C_BNtGV3sVvZCnNZaER9rstmy0s)
- [Multiprocessing](https://colab.research.google.com/drive/1ZzNFMUUi923foaVsYb4YjPy4mjKtnOxb)
- [Monitor Training and Plotting](https://colab.research.google.com/drive/1L_IMo6v0a0ALK8nefZm6PqPSy0vZIWBT)
- [Atari Games](https://colab.research.google.com/drive/1iYK11yDzOOqnrXi1Sfjm1iekZr4cxLaN)


## Implemented Algorithms

| **Name**            | **Refactored**<sup>(1)</sup> | **Recurrent**      | ```Box```          | ```Discrete```     | ```MultiDiscrete``` | ```MultiBinary```  | **Multi Processing**              |
| ------------------- | ---------------------------- | ------------------ | ------------------ | ------------------ | ------------------- | ------------------ | --------------------------------- |
| A2C                 | :heavy_check_mark:           | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark:  | :heavy_check_mark: | :heavy_check_mark:                |
| ACER                | :heavy_check_mark:           | :heavy_check_mark: | :x: <sup>(5)</sup> | :heavy_check_mark: | :x:                 | :x:                | :heavy_check_mark:                |
| ACKTR               | :heavy_check_mark:           | :heavy_check_mark: | :x: <sup>(5)</sup> | :heavy_check_mark: | :x:                 | :x:                | :heavy_check_mark:                |
| DDPG                | :heavy_check_mark:           | :x:                | :heavy_check_mark: | :x:                | :x:                 | :x:                | :x:                               |
| DQN                 | :heavy_check_mark:           | :x:                | :x:                | :heavy_check_mark: | :x:                 | :x:                | :x:                               |
| GAIL <sup>(2)</sup> | :heavy_check_mark:           | :heavy_check_mark: | :heavy_check_mark: | :x:                | :x:                 | :x:                | :heavy_check_mark: <sup>(4)</sup> |
| HER <sup>(3)</sup>  | :x: <sup>(5)</sup>           | :x:                | :heavy_check_mark: | :x:                | :x:                 | :x:                | :x:                               |
| PPO1                | :heavy_check_mark:           | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark:  | :heavy_check_mark: | :heavy_check_mark: <sup>(4)</sup> |
| PPO2                | :heavy_check_mark:           | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark:  | :heavy_check_mark: | :heavy_check_mark:                |
| TRPO                | :heavy_check_mark:           | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark:  | :heavy_check_mark: | :heavy_check_mark: <sup>(4)</sup> |

<sup><sup>(1): Whether or not the algorithm has be refactored to fit the ```BaseRLModel``` class.</sup></sup><br>
<sup><sup>(2): Only implemented for TRPO.</sup></sup><br>
<sup><sup>(3): Only implemented for DDPG.</sup></sup><br>
<sup><sup>(4): Multi Processing with [MPI](https://mpi4py.readthedocs.io/en/stable/).</sup></sup><br>
<sup><sup>(5): TODO, in project scope.</sup></sup>

Actions ```gym.spaces```:
 * ```Box```: A N-dimensional box that containes every point in the action space.
 * ```Discrete```: A list of possible actions, where each timestep only one of the actions can be used.
 * ```MultiDiscrete```: A list of possible actions, where each timestep only one action of each discrete set can be used.
 * ```MultiBinary```: A list of possible actions, where each timestep any of the actions can be used in any combination.


## MuJoCo
Some of the baselines examples use [MuJoCo](http://www.mujoco.org) (multi-joint dynamics in contact) physics simulator, which is proprietary and requires binaries and a license (temporary 30-day license can be obtained from [www.mujoco.org](http://www.mujoco.org)). Instructions on setting up MuJoCo can be found [here](https://github.com/openai/mujoco-py)

## Testing the installation
All unit tests in baselines can be run using pytest runner:
```
pip install pytest pytest-cov
pytest --cov-config .coveragerc --cov-report html --cov-report term --cov=.
```

## Citing the Project

To cite this repository in publications:

```
    @misc{stable-baselines,
      author = {Hill, Ashley and Raffin, Antonin and Traore, Rene and Dhariwal, Prafulla and Hesse, Christopher and Klimov, Oleg and Nichol, Alex and Plappert, Matthias and Radford, Alec and Schulman, John and Sidor, Szymon and Wu, Yuhuai},
      title = {Stable Baselines},
      year = {2018},
      publisher = {GitHub},
      journal = {GitHub repository},
      howpublished = {\url{https://github.com/hill-a/stable-baselines}},
    }
```

## Maintainers

Stable-Baselines is currently maintained by [Ashley Hill](https://github.com/hill-a) (aka @hill-a) and [Antonin Raffin](https://github.com/araffin) (aka @araffin).

## How To Contribute

To any interested in making the baselines better, there is still some documentation that needs to be done.
If you want to contribute, please open an issue first and then propose your pull request.

Nice to have (for the future):
- [ ] Continuous actions support for ACER
- [ ] Continuous actions support for ACKTR

## Acknowledgments

Stable Baselines was created in the [robotics lab U2IS](http://u2is.ensta-paristech.fr/index.php?lang=en) ([INRIA Flowers](https://flowers.inria.fr/) team) at [ENSTA ParisTech](http://www.ensta-paristech.fr/en).

Logo credits: L.M. Tenkes
