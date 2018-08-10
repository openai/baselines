<img src="data/logo.jpg" width=25% align="right" /> [![Build Status](https://travis-ci.org/hill-a/stable-baselines.svg?branch=master)](https://travis-ci.org/hill-a/stable-baselines) [![Codacy Badge](https://api.codacy.com/project/badge/Grade/3bcb4cd6d76a4270acb16b5fe6dd9efa)](https://www.codacy.com/app/baselines_janitors/stable-baselines?utm_source=github.com&amp;utm_medium=referral&amp;utm_content=hill-a/stable-baselines&amp;utm_campaign=Badge_Grade) [![Codacy Badge](https://api.codacy.com/project/badge/Coverage/3bcb4cd6d76a4270acb16b5fe6dd9efa)](https://www.codacy.com/app/baselines_janitors/stable-baselines?utm_source=github.com&utm_medium=referral&utm_content=hill-a/stable-baselines&utm_campaign=Badge_Coverage)

# Baselines

OpenAI Baselines is a set of high-quality implementations of reinforcement learning algorithms.

These algorithms will make it easier for the research community to replicate, refine, and identify new ideas, and will create good baselines to build research on top of. Our DQN implementation and its variants are roughly on par with the scores in published papers. We expect they will be used as a base around which new ideas can be added, and as a tool for comparing a new approach against existing ones.

## Usage

### Implemented Algorithms

| **Name**            | **refactored**<sup>(1)</sup> | **Reccurent**      | **Actions** ```Box``` |  **Actions** ```Discrete``` |  **Actions** ```MultiDiscrete``` |  **Actions** ```MultiBinary```|
| ------------------- | ---------------------------- | ------------------ | --------------------- | --------------------------- | -------------------------------- | ----------------------------- |
| A2C                 | :heavy_check_mark:           | :heavy_check_mark: | :heavy_check_mark:    | :heavy_check_mark:          | :heavy_check_mark:               | :heavy_check_mark:            |
| ACER                | :heavy_check_mark:           | :heavy_check_mark: | :x: <sup>(4)</sup>    | :heavy_check_mark:          | :x:                              | :x:                           |
| ACKTR               | :heavy_check_mark:           | :heavy_check_mark: | :x: <sup>(4)</sup>    | :heavy_check_mark:          | :x:                              | :x:                           |
| DDPG                | :heavy_check_mark:           | :x:                | :heavy_check_mark:    | :x:                         | :x:                              | :x:                           |
| DeepQ               | :heavy_check_mark:           | :x:                | :x:                   | :heavy_check_mark:          | :x:                              | :x:                           |
| GAIL <sup>(2)</sup> | :heavy_check_mark:           | :heavy_check_mark: | :heavy_check_mark:    | :heavy_check_mark:          | :heavy_check_mark:               | :heavy_check_mark:            |
| HER <sup>(3)</sup>  | :x:                          | :x:                | :heavy_check_mark:    | :x:                         | :x:                              | :x:                           |
| PPO1                | :heavy_check_mark:           | :heavy_check_mark: | :heavy_check_mark:    | :heavy_check_mark:          | :heavy_check_mark:               | :heavy_check_mark:            |
| PPO2                | :heavy_check_mark:           | :heavy_check_mark: | :heavy_check_mark:    | :heavy_check_mark:          | :heavy_check_mark:               | :heavy_check_mark:            |
| TRPO                | :heavy_check_mark:           | :heavy_check_mark: | :heavy_check_mark:    | :heavy_check_mark:          | :heavy_check_mark:               | :heavy_check_mark:            |

<sup><sup>(1): Whether or not the algorithm has be refactored to fit the ```BaseRLModel``` class.</sup></sup><br>
<sup><sup>(2): Only implemented for TRPO.</sup></sup><br>
<sup><sup>(3): Only implemented for DDPG.</sup></sup><br>
<sup><sup>(4): TODO, in project scope.</sup></sup>

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

from baselines.common.policies import MlpPolicy, MlpLstmPolicy, MlpLnLstmPolicy, \
    CnnPolicy, CnnLstmPolicy, CnnLnLstmPolicy
from baselines.common.vec_env.dummy_vec_env import DummyVecEnv
from baselines.acktr import ACKTR

env = gym.make('CartPole-v0')
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
import gym

from baselines.common.policies import MlpPolicy, MlpLstmPolicy, MlpLnLstmPolicy, \
    CnnPolicy, CnnLstmPolicy, CnnLnLstmPolicy
from baselines.common.vec_env.dummy_vec_env import DummyVecEnv
from baselines.a2c import A2C

env = gym.make('Breakout-v0')
env = DummyVecEnv([lambda: env])

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
import gym

from baselines.common.vec_env.dummy_vec_env import DummyVecEnv
from baselines.deepq import DeepQ, models

env = gym.make('MsPacman-v0')
env = DummyVecEnv([lambda: env])

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
import gym

from baselines.common.policies import MlpPolicy, MlpLstmPolicy, MlpLnLstmPolicy, \
    CnnPolicy, CnnLstmPolicy, CnnLnLstmPolicy
from baselines.common.vec_env.dummy_vec_env import DummyVecEnv
from baselines.ppo2 import PPO2

env = gym.make('DemonAttack-v0')
env = DummyVecEnv([lambda: env])

model = PPO2(CnnPolicy, env, verbose=1)
model.learn(10000)

obs = env.reset()
for i in range(1000):
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    env.render()


env = gym.make('SpaceInvaders-v0')
env = DummyVecEnv([lambda: env])

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
Clone the repo and cd into it:
```bash
git clone https://github.com/openai/baselines.git
cd baselines
```
If using virtualenv, create a new virtualenv and activate it
```bash
    virtualenv env --python=python3
    . env/bin/activate
```
Install baselines package
```bash
pip install -e .
```
### MuJoCo
Some of the baselines examples use [MuJoCo](http://www.mujoco.org) (multi-joint dynamics in contact) physics simulator, which is proprietary and requires binaries and a license (temporary 30-day license can be obtained from [www.mujoco.org](http://www.mujoco.org)). Instructions on setting up MuJoCo can be found [here](https://github.com/openai/mujoco-py)

## Testing the installation
All unit tests in baselines can be run using pytest runner:
```
pip install pytest
pytest
```

## Subpackages

- [A2C](baselines/a2c)
- [ACER](baselines/acer)
- [ACKTR](baselines/acktr)
- [DDPG](baselines/ddpg)
- [DQN](baselines/deepq)
- [GAIL](baselines/gail)
- [HER](baselines/her)
- [PPO1](baselines/ppo1) (Multi-CPU using MPI)
- [PPO2](baselines/ppo2) (Optimized for GPU)
- [TRPO](baselines/trpo_mpi)

To cite this repository in publications:

    @misc{baselines,
      author = {Dhariwal, Prafulla and Hesse, Christopher and Klimov, Oleg and Nichol, Alex and Plappert, Matthias and Radford, Alec and Schulman, John and Sidor, Szymon and Wu, Yuhuai},
      title = {OpenAI Baselines},
      year = {2017},
      publisher = {GitHub},
      journal = {GitHub repository},
      howpublished = {\url{https://github.com/openai/baselines}},
    }
