[![Build Status](https://travis-ci.org/hill-a/stable-baselines.svg?branch=master)](https://travis-ci.org/hill-a/stable-baselines) [![Codacy Badge](https://api.codacy.com/project/badge/Grade/3bcb4cd6d76a4270acb16b5fe6dd9efa)](https://www.codacy.com/app/baselines_janitors/stable-baselines?utm_source=github.com&amp;utm_medium=referral&amp;utm_content=hill-a/stable-baselines&amp;utm_campaign=Badge_Grade) [![Codacy Badge](https://api.codacy.com/project/badge/Coverage/3bcb4cd6d76a4270acb16b5fe6dd9efa)](https://www.codacy.com/app/baselines_janitors/stable-baselines?utm_source=github.com&utm_medium=referral&utm_content=hill-a/stable-baselines&utm_campaign=Badge_Coverage)

# Stable Baselines

Stable Baselines is a set of improved implementations of reinforcement learning algorithms based on OpenAI [Baselines](https://github.com/openai/baselines/).

These algorithms will make it easier for the research community and industry to replicate, refine, and identify new ideas, and will create good baselines to build projects on top of. We expect these tools will be used as a base around which new ideas can be added, and as a tool for comparing a new approach against existing ones. We also hope that the simplicity of these tools will allow beginers to experiment with a more advanced toolset, without being buried in implementation details.

## Main differences with OpenAI Baselines

This toolset is a fork of OpenAI Baselines, with a major strutural refactoring, and code cleanups:
- Unified structure for all algorithms
- PEP8 compliant (unified code style)
- Documented functions and classes
- More tests & more code coverage

## Usage

### Getting started

Most of the library tries to follow a sklearn-like syntax for the Reinforcement Learning algorithms.  

Here is a quick example of how to train and run PPO2 on a cartpole environment:
```python
import gym

from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env.dummy_vec_env import DummyVecEnv
from stable_baselines.ppo2 import PPO2

env = gym.make('CartPole-v0')
env = DummyVecEnv([lambda: env])  # The algorithms require a vectorized environment to run

model = PPO2(MlpPolicy, env, verbose=1)
model.learn(10000)

obs = env.reset()
for i in range(1000):
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    env.render()
```

### Implemented Algorithms

| **Name**            | **refactored**<sup>(1)</sup> | **Reccurent**      | ```Box```          | ```Discrete```     | ```MultiDiscrete``` | ```MultiBinary```  | **Multi Processing**              |
| ------------------- | ---------------------------- | ------------------ | ------------------ | ------------------ | ------------------- | ------------------ | --------------------------------- |
| A2C                 | :heavy_check_mark:           | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark:  | :heavy_check_mark: | :heavy_check_mark:                |
| ACER                | :heavy_check_mark:           | :heavy_check_mark: | :x: <sup>(5)</sup> | :heavy_check_mark: | :x:                 | :x:                | :heavy_check_mark:                |
| ACKTR               | :heavy_check_mark:           | :heavy_check_mark: | :x: <sup>(5)</sup> | :heavy_check_mark: | :x:                 | :x:                | :heavy_check_mark:                |
| DDPG                | :heavy_check_mark:           | :x:                | :heavy_check_mark: | :x:                | :x:                 | :x:                | :x:                               |
| DeepQ               | :heavy_check_mark:           | :x:                | :x:                | :heavy_check_mark: | :x:                 | :x:                | :x:                               |
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

### Examples

here are a few barebones examples of how to use this library:

ACKTR with ```CartPole-v0```:
```python
import gym

from stable_baselines.common.policies import MlpPolicy, MlpLstmPolicy, MlpLnLstmPolicy, \
    CnnPolicy, CnnLstmPolicy, CnnLnLstmPolicy
from stable_baselines.common.vec_env.dummy_vec_env import DummyVecEnv
from stable_baselines.acktr import ACKTR

env = gym.make('CartPole-v0')
# Vectorize the environment, as some models requires it. But all the models can use vectorized environments
env = DummyVecEnv([lambda: env])

model = ACKTR(MlpPolicy, env, gamma=0.5, verbose=1)
model.learn(25000)
model.save("acktr_env")

obs = env.reset()
while True:
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    env.render()
```

A2C with ```Breakout-v0```:
```python
from stable_baselines.common.cmd_util import make_atari_env
from stable_baselines.common.policies import MlpPolicy, MlpLstmPolicy, MlpLnLstmPolicy, \
    CnnPolicy, CnnLstmPolicy, CnnLnLstmPolicy
from stable_baselines.common.vec_env.dummy_vec_env import DummyVecEnv
from stable_baselines.a2c import A2C

# There already exists an environment generator that will make and wrap atari environments correctly.
env = make_atari_env('BreakoutNoFrameskip-v4', num_env=8, seed=0)

model = A2C(CnnPolicy, env, verbose=1)
model.learn(25000)
model.save("a2c_env")

obs = env.reset()
while True:
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    env.render()
```

DeepQ with ```MsPacman-v0```:
```python
from stable_baselines.common.cmd_util import make_atari_env
from stable_baselines.common.vec_env.dummy_vec_env import DummyVecEnv
from stable_baselines.deepq import DeepQ, models

# There already exists an environment generator that will make and wrap atari environments correctly
# Here we set the num_env to 1, as DeepQ does not support multi-environments
env = make_atari_env('MsPacmanNoFrameskip-v4', num_env=1, seed=0)

# Here deepq does not use the standard Actor-Critic policies
model = DeepQ(models.cnn_to_mlp(convs=[(32, 8, 4), (64, 4, 2), (64, 3, 1)], hiddens=[64]), env, verbose=1)
model.learn(5000)
model.save("deepq_env")

obs = env.reset()
while True:
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    env.render()
```


You can also move from one environment to an other for continuous learning (PPO2 on ```DemonAttack-v0```,  then transfered on ```SpaceInvaders-v0```):
```python
from stable_baselines.common.cmd_util import make_atari_env
from stable_baselines.common.policies import MlpPolicy, MlpLstmPolicy, MlpLnLstmPolicy, \
    CnnPolicy, CnnLstmPolicy, CnnLnLstmPolicy
from stable_baselines.common.vec_env.dummy_vec_env import DummyVecEnv
from stable_baselines.ppo2 import PPO2

# There already exists an environment generator that will make and wrap atari environments correctly
env = make_atari_env('DemonAttackNoFrameskip-v4', num_env=8, seed=0)

model = PPO2(CnnPolicy, env, verbose=1)
model.learn(10000)

obs = env.reset()
for i in range(1000):
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    env.render()

# The number of environments must be identical when changing environments
env = make_atari_env('SpaceInvadersNoFrameskip-v4', num_env=8, seed=0)

# change env
model.set_env(env)
model.learn(10000)

obs = env.reset()
while True:
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    env.render()
```

## Prerequisites
Baselines requires python3 (>=3.5) with the development headers. You'll also need system packages CMake, OpenMPI and zlib. Those can be installed as follows
### Ubuntu

```bash
sudo apt-get update && sudo apt-get install cmake libopenmpi-dev python3-dev zlib1g-dev
```

### Mac OS X
Installation of system packages on Mac requires [Homebrew](https://brew.sh). With Homebrew installed, run the follwing:
```bash
brew install cmake openmpi
```

## Virtual environment
From the general python package sanity perspective, it is a good idea to use virtual environments (virtualenvs) to make sure packages from different projects do not interfere with each other. You can install virtualenv (which is itself a pip package) via
```bash
pip install virtualenv
```
Virtualenvs are essentially folders that have copies of python executable and all python packages.
To create a virtualenv called venv with python3, one runs
```bash
virtualenv /path/to/venv --python=python3
```
To activate a virtualenv:
```
. /path/to/venv/bin/activate
```
More thorough tutorial on virtualenvs and options can be found [here](https://virtualenv.pypa.io/en/stable/)


## Installation
Install the Stable Baselines package
```bash
pip install git+https://github.com/hill-a/stable-baselines
```
### MuJoCo
Some of the baselines examples use [MuJoCo](http://www.mujoco.org) (multi-joint dynamics in contact) physics simulator, which is proprietary and requires binaries and a license (temporary 30-day license can be obtained from [www.mujoco.org](http://www.mujoco.org)). Instructions on setting up MuJoCo can be found [here](https://github.com/openai/mujoco-py)

## Testing the installation
All unit tests in baselines can be run using pytest runner:
```
pip install pytest pytest-cov
pytest --cov-config .coveragerc --cov-report html --cov-report term --cov=.
```

## Subpackages

- [A2C](stable_baselines/a2c)
- [ACER](stable_baselines/acer)
- [ACKTR](stable_baselines/acktr)
- [DDPG](stable_baselines/ddpg)
- [DQN](stable_baselines/deepq)
- [GAIL](stable_baselines/gail)
- [HER](stable_baselines/her)
- [PPO1](stable_baselines/ppo1) (Multi-CPU using MPI)
- [PPO2](stable_baselines/ppo2) (Optimized for GPU)
- [TRPO](stable_baselines/trpo_mpi)
