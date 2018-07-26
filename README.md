<img src="data/logo.jpg" width=25% align="right" /> [![Build status](https://travis-ci.org/openai/baselines.svg?branch=master)](https://travis-ci.org/openai/baselines)

# Baselines

OpenAI Baselines is a set of high-quality implementations of reinforcement learning algorithms.

These algorithms will make it easier for the research community to replicate, refine, and identify new ideas, and will create good baselines to build research on top of. Our DQN implementation and its variants are roughly on par with the scores in published papers. We expect they will be used as a base around which new ideas can be added, and as a tool for comparing a new approach against existing ones. 

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

## Testing the installation
All unit tests in baselines can be run using pytest runner:
```
pip install pytest
pytest
```

## Training models
Most of the algorithms in baselines repo are used as follows:
```bash
    python -m baselines.common.run --alg=<name of the algorithm> --env=<environment_id> [additional arguments]
```
### Example 1. PPO with MuJoCo Humanoid
For instance, to train a fully-connected network controlling MuJoCo humanoid using ppo2 for 20M timesteps
```bash
    python -m baselines.common.run --alg=ppo2 --env=Humanoid-v2 --network=mlp --num-timesteps=2e7
```
Note that for mujoco environments fully-connected network is default, so we can omit `--network=mlp`
The hyperparameters for both network and the learning algorithm can be controlled via the command line, for instance:
```bash
    python -m baselines.common.run --alg=ppo2 --env=Humanoid-v2 --network=mlp --num-timesteps=2e7 --ent_coef=0.1 --num_hidden=32 --num_layers=3
```
will set entropy coeffient to 0.1, and construct fully connected network with 3 layers with 32 hidden units in each. 
See docstrings in [common/models.py](common/models.py) for description of network parameters for each type of model, and 
docstring for [baselines/ppo2/ppo2.py/learn()](ppo2/ppo2.py) fir the description of the ppo2 hyperparamters. 

### Example 2. DQN on Atari 
DQN with Atari is at this point a classics of benchmarks. To run the baselines implementation of DQN on Atari Pong:
```
    python -m baselines.common.run --alg=deepq --env=PongNoFrameskip-v4 --num-timesteps=10000000
```

## Saving, loading and visualizing models
The algorithms serialization API is not properly unified yet; however, there is a simple method to save / restore trained models. 
`--model-path` command-line option loads the tensorflow state from a given path before training, and saves it after the training. 
Let's imagine you'd like to train a2c on MuJoCo humanoid, save the model and then later visualize what has it learnt. 
```bash
    python -m baselines.common.run --alg=a2c --env=Humanoid-v2 --num-timesteps=2e7 --model-path=~/models/humanoid_20M_a2c
```
To load and visualize the model, we'll do the following - load the model, train it for trivial number of steps (say, 10), and then visualize: 
```bash
    python -m baselines.common.run --alg=a2c --env=Humanoid-v2 --num-timesteps=10 --model-path=~/models/humanoid_20M_a2c --play
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
