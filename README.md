[![Build Status](https://travis-ci.com/hill-a/stable-baselines.svg?branch=stable)](https://travis-ci.com/hill-a/stable-baselines) [![Codacy Badge](https://api.codacy.com/project/badge/Grade/3bcb4cd6d76a4270acb16b5fe6dd9efa)](https://www.codacy.com/app/baselines_janitors/stable-baselines?utm_source=github.com&amp;utm_medium=referral&amp;utm_content=hill-a/stable-baselines&amp;utm_campaign=Badge_Grade) [![Codacy Badge](https://api.codacy.com/project/badge/Coverage/3bcb4cd6d76a4270acb16b5fe6dd9efa)](https://www.codacy.com/app/baselines_janitors/stable-baselines?utm_source=github.com&utm_medium=referral&utm_content=hill-a/stable-baselines&utm_campaign=Badge_Coverage)

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


Table of Contents
=================
* [Main differences with OpenAI Baselines](#main-differences-with-openai-baselines)
* [Usage](#usage)
  * [Getting started](#getting-started)
  * [Try it online with Colab Notebooks \!](#try-it-online-with-colab-notebooks-)
  * [Implemented Algorithms](#implemented-algorithms)
  * [Examples](#examples)
    * [ACKTR with CartPole\-v1](#acktr-with-cartpole-v1)
    * [A2C with Breakout\-v0 (Atari)](#a2c-with-breakout-v0-atari)
    * [DQN with MsPacman\-v0](#dqn-with-mspacman-v0)
    * [Multiprocessing: Unleashing the Power of Vectorized Environments](#multiprocessing-unleashing-the-power-of-vectorized-environments)
    * [Using Callback: Monitoring Training](#using-callback-monitoring-training)
    * [Custom Policies](#custom-policies)
    * [Normalize Input Features](#normalize-input-features)
    * [Continual Learning](#continual-learning)
* [Prerequisites](#prerequisites)
  * [Ubuntu](#ubuntu)
  * [Mac OS X](#mac-os-x)
* [Virtual environment](#virtual-environment)
* [Installation](#installation)
  * [MuJoCo](#mujoco)
* [Testing the installation](#testing-the-installation)
* [Subpackages](#subpackages)
* [How To Contribute](#how-to-contribute)
* [Bonus](#bonus)


## Usage

### Getting started

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

Or just train a model with a one liner if [the environment is registed in Gym](https://github.com/openai/gym/wiki/Environments):
```python
from stable_baselines.common.policies import MlpPolicy
from stable_baselines import PPO2

model = PPO2(MlpPolicy, 'CartPole-v1').learn(10000)
```

### Try it online with Colab Notebooks !

All the following examples can be executed online using Google colab notebooks:

- [Getting Started](https://colab.research.google.com/drive/1_1H5bjWKYBVKbbs-Kj83dsfuZieDNcFU)
- [Training, Saving, Loading](https://colab.research.google.com/drive/1KoAQ1C_BNtGV3sVvZCnNZaER9rstmy0s)
- [Multiprocessing](https://colab.research.google.com/drive/1ZzNFMUUi923foaVsYb4YjPy4mjKtnOxb)
- [Monitor Training and Plotting](https://colab.research.google.com/drive/1L_IMo6v0a0ALK8nefZm6PqPSy0vZIWBT)
- [Atari Games](https://colab.research.google.com/drive/1iYK11yDzOOqnrXi1Sfjm1iekZr4cxLaN)


### Implemented Algorithms

| **Name**            | **Refactored**<sup>(1)</sup> | **Recurrent**      | ```Box```          | ```Discrete```     | ```MultiDiscrete``` | ```MultiBinary```  | **Multi Processing**              |
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

#### ACKTR with ```CartPole-v1```
```python
import gym

from stable_baselines.common.policies import MlpPolicy, MlpLstmPolicy, MlpLnLstmPolicy, \
    CnnPolicy, CnnLstmPolicy, CnnLnLstmPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import ACKTR

env = gym.make('CartPole-v1')
# Vectorize the environment, as some models requires it.
# But all the models can use vectorized environments
env = DummyVecEnv([lambda: env])

model = ACKTR(MlpPolicy, env, gamma=0.5, verbose=1)
model.learn(total_timesteps=25000)
model.save(save_path="acktr_env")

obs = env.reset()
while True:
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    env.render()
```

#### A2C with ```Breakout-v0``` (Atari)

Training a RL agent on Atari games is straightforward thanks to `make_atari_env` helper function. It will do all the [preprocessing](https://danieltakeshi.github.io/2016/11/25/frame-skipping-and-preprocessing-for-deep-q-networks-on-atari-2600-games/) and multiprocessing for you.

```python
from stable_baselines.common.cmd_util import make_atari_env
from stable_baselines.common.policies import MlpPolicy, MlpLstmPolicy, MlpLnLstmPolicy, \
    CnnPolicy, CnnLstmPolicy, CnnLnLstmPolicy
from stable_baselines.common.vec_env import DummyVecEnv, VecFrameStack
from stable_baselines import A2C

# There already exists an environment generator that will make and wrap atari environments correctly.
env = make_atari_env('BreakoutNoFrameskip-v4', num_env=8, seed=0)
# Stack 4 frames
env = VecFrameStack(env, n_stack=4)

model = A2C(CnnPolicy, env, verbose=1)
model.learn(total_timesteps=25000)
model.save(save_path="a2c_env")

obs = env.reset()
while True:
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    env.render()
```

#### DQN with ```MsPacman-v0```
```python
from stable_baselines.common.cmd_util import make_atari_env
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import DeepQ, models

# There already exists an environment generator that will make and wrap atari environments correctly
# Here we set the num_env to 1, as DeepQ does not support multi-environments
env = make_atari_env('MsPacmanNoFrameskip-v4', num_env=1, seed=0)

# Here deepq does not use the standard Actor-Critic policies
model = DeepQ(models.cnn_to_mlp(convs=[(32, 8, 4), (64, 4, 2), (64, 3, 1)], hiddens=[64]), env, verbose=1)
model.learn(total_timesteps=5000)
model.save(save_path="deepq_env")

obs = env.reset()
while True:
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    env.render()
```

#### Multiprocessing: Unleashing the Power of Vectorized Environments

```python
import gym
import numpy as np

from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import SubprocVecEnv
from stable_baselines.common import set_global_seeds
from stable_baselines import ACKTR

def make_env(env_id, rank, seed=0):
    """
    Utility function for multiprocessed env.

    :param env_id: (str) the environment ID
    :param num_env: (int) the number of environment you wish to have in subprocesses
    :param seed: (int) the inital seed for RNG
    :param rank: (int) index of the subprocess
    """
    def _init():
        env = gym.make(env_id)
        env.seed(seed + rank)
        return env
    set_global_seeds(seed)
    return _init

env_id = "CartPole-v1"
num_cpu = 4  # Number of processes to use
# Create the vectorized environment
env = SubprocVecEnv([make_env(env_id, i) for i in range(num_cpu)])

model = ACKTR(MlpPolicy, env, verbose=1)
model.learn(total_timesteps=25000)

obs = env.reset()
for _ in range(1000):
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    env.render()
```

#### Using Callback: Monitoring Training

You can define a custom callback function that will be called inside the agent. This could be useful when you want to monitor training, for instance display live learning curve in Tensorboard (or in Visdom) or save the best agent.

```python
import os

import gym
import numpy as np
import matplotlib.pyplot as plt

from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env.dummy_vec_env import DummyVecEnv
from stable_baselines.bench import Monitor
from stable_baselines.results_plotter import load_results, ts2xy
from stable_baselines import DDPG
from stable_baselines.ddpg.noise import AdaptiveParamNoiseSpec


best_mean_reward, n_steps = -np.inf, 0

def callback(_locals, _globals):
    """
    Callback called at each step (for DQN an others) or after n steps (see ACER or PPO2)
    :param _locals: (dict)
    :param _globals: (dict)
    """
    global n_steps, best_mean_reward
    # Print stats every 1000 calls
    if (n_steps + 1) % 1000 == 0:
        # Evaluate policy performance
        x, y = ts2xy(load_results(log_dir), 'timesteps')
        if len(x) > 0:
            mean_reward = np.mean(y[-100:])
            print(x[-1], 'timesteps')
            print("Best mean reward: {:.2f} - Last mean reward per episode: {:.2f}".format(best_mean_reward, mean_reward))

            # New best model, you could save the agent here
            if mean_reward > best_mean_reward:
                best_mean_reward = mean_reward
                # Example for saving best model
                print("Saving new best model")
                _locals['self'].save(log_dir + 'best_model.pkl')
    n_steps += 1
    return False


# Create log dir
log_dir = "/tmp/gym/"
os.makedirs(log_dir, exist_ok=True)

# Create and wrap the environment
env = gym.make('LunarLanderContinuous-v2')
env = Monitor(env, log_dir, allow_early_resets=True)
env = DummyVecEnv([lambda: env])

# Add some param noise for exploration
param_noise = AdaptiveParamNoiseSpec(initial_stddev=0.2, desired_action_stddev=0.2)
model = DDPG(MlpPolicy, env, param_noise=param_noise, memory_limit=int(1e6), verbose=0)
# Train the agent
model.learn(total_timesteps=200000, callback=callback)

```

#### Custom Policies

You can also make custom policies to train with:
```python
import gym

from stable_baselines.common.policies import FeedForwardPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import ppo2

# Custom MLP policy of 3 layers of 256, 64 and 16
class CustomPolicy(FeedForwardPolicy):
    def __init__(self, *args, **kwargs):
        super(CustomPolicy, self).__init__(*args, **kwargs, layers=[256, 64, 16], feature_extraction="mlp")

env = gym.make('LunarLander-v2')
env = DummyVecEnv([lambda: env])

model = PPO2(CustomPolicy, env, verbose=1)
model.learn(total_timesteps=100000)

obs = env.reset()
while True:
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    env.render()
```

#### Normalize Input Features

By default, images are scaled (dividing by the maximum value definied by the environment) but not other type of input.
For that, a wrapper exists and will compute a running average and standard deviation of input features (it can do the same for rewards).

```python
import gym

from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines import PPO2

env = DummyVecEnv([lambda: gym.make("Reacher-v2")])
# Automatically normalize the input features
env = VecNormalize(env, norm_obs=True, norm_reward=False,
                   clip_obs=10.)

model = PPO2(MlpPolicy, env)
model.learn(total_timesteps=2000)

# Don't forget to save the running average when saving the agent
log_dir = "/tmp/"
model.save(log_dir + "ppo_reacher")
env.save_running_average(log_dir)
```


#### Continual Learning

You can also move from one environment to an other for continuous learning (PPO2 on ```DemonAttack-v0```,  then transferred on ```SpaceInvaders-v0```):
```python
from stable_baselines.common.cmd_util import make_atari_env
from stable_baselines.common.policies import MlpPolicy, MlpLstmPolicy, MlpLnLstmPolicy, \
    CnnPolicy, CnnLstmPolicy, CnnLnLstmPolicy
from stable_baselines import PPO2

# There already exists an environment generator that will make and wrap atari environments correctly
env = make_atari_env('DemonAttackNoFrameskip-v4', num_env=8, seed=0)

model = PPO2(CnnPolicy, env, verbose=1)
model.learn(total_timesteps=10000)

obs = env.reset()
for i in range(1000):
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    env.render()

# The number of environments must be identical when changing environments
env = make_atari_env('SpaceInvadersNoFrameskip-v4', num_env=8, seed=0)

# change env
model.set_env(env)
model.learn(total_timesteps=10000)

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

Using pip from pypi:
```
pip install stable-baselines
```

From source:
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

## How To Contribute

To any interested in making the baselines better, there is still some documentation that needs to be done.
If you want to contribute, please open an issue first and then propose your pull request.

Nice to have (for the future):
- [ ] Continuous actions support for ACER
- [ ] Continuous actions support for ACKTR
- [ ] Html documentation (see issue #11)

## Bonus

Make a gif of a trained agent (you need to install imageio):

```python
import imageio
import numpy as np

from stable_baselines.common.policies import MlpPolicy
from stable_baselines import A2C

model = A2C(MlpPolicy, "LunarLander-v2").learn(100000)

images = []                                                                                              
obs = model.env.reset()
img = model.env.render(mode='rgb_array')
for i in range(350):
    images.append(img)
    action, _ = model.predict(obs)
    obs, _, _ ,_ = model.env.step(action)
    img = model.env.render(mode='rgb_array')

imageio.mimsave('lander_a2c.gif', [np.array(img[0]) for i, img in enumerate(images) if i%2 == 0], fps=29)
```
