import sys
import subprocess
from setuptools import setup, find_packages
from distutils.version import LooseVersion

if sys.version_info.major != 3:
    print('This Python is only compatible with Python 3, but you are running '
          'Python {}. The installation will likely fail.'.format(sys.version_info.major))

# Check tensorflow installation to avoid
# breaking pre-installed tf gpu
install_tf, tf_gpu = False, False
try:
    import tensorflow as tf
    if tf.__version__ < LooseVersion('1.5.0'):
        install_tf = True
        # check if a gpu version is needed
        tf_gpu = tf.test.is_gpu_available()
except ImportError:
    install_tf = True
    # Check if a nvidia gpu is present
    for command in ['nvidia-smi', '/usr/bin/nvidia-smi', 'nvidia-smi.exe']:
        try:
            if subprocess.call([command]) == 0:
                tf_gpu = True
                break
        except IOError:  # command does not exist / is not executable
            pass

tf_dependency = []
if install_tf:
    tf_dependency = ['tensorflow-gpu>=1.5.0'] if tf_gpu else ['tensorflow>=1.5.0']
    if tf_gpu:
        print("A GPU was detected, tensorflow-gpu will be installed")


long_description = """
[![Build Status](https://travis-ci.com/hill-a/stable-baselines.svg?branch=master)](https://travis-ci.com/hill-a/stable-baselines) [![Documentation Status](https://readthedocs.org/projects/stable-baselines/badge/?version=master)](https://stable-baselines.readthedocs.io/en/master/?badge=master) [![Codacy Badge](https://api.codacy.com/project/badge/Grade/3bcb4cd6d76a4270acb16b5fe6dd9efa)](https://www.codacy.com/app/baselines_janitors/stable-baselines?utm_source=github.com&amp;utm_medium=referral&amp;utm_content=hill-a/stable-baselines&amp;utm_campaign=Badge_Grade) [![Codacy Badge](https://api.codacy.com/project/badge/Coverage/3bcb4cd6d76a4270acb16b5fe6dd9efa)](https://www.codacy.com/app/baselines_janitors/stable-baselines?utm_source=github.com&utm_medium=referral&utm_content=hill-a/stable-baselines&utm_campaign=Badge_Coverage)

# Stable Baselines

Stable Baselines is a set of improved implementations of reinforcement learning algorithms based on OpenAI [Baselines](https://github.com/openai/baselines/).

These algorithms will make it easier for the research community and industry to replicate, refine, and identify new ideas, and will create good baselines to build projects on top of. We expect these tools will be used as a base around which new ideas can be added, and as a tool for comparing a new approach against existing ones. We also hope that the simplicity of these tools will allow beginners to experiment with a more advanced toolset, without being buried in implementation details.

## Main differences with OpenAI Baselines
This toolset is a fork of OpenAI Baselines, with a major structural refactoring, and code cleanups:

-   Unified structure for all algorithms
-   PEP8 compliant (unified code style)
-   Documented functions and classes
-   More tests & more code coverage

## Links

Repository:
https://github.com/hill-a/stable-baselines

Medium article:
https://medium.com/@araffin/df87c4b2fc82

Documentation:
https://stable-baselines.readthedocs.io/en/master/

RL Baselines Zoo:
https://github.com/araffin/rl-baselines-zoo

## Quick example

Most of the library tries to follow a sklearn-like syntax for the Reinforcement Learning algorithms using Gym.

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

"""

setup(name='stable_baselines',
      packages=[package for package in find_packages()
                if package.startswith('stable_baselines')],
      install_requires=[
          'gym[atari,classic_control]>=0.10.9',
          'scipy',
          'tqdm',
          'joblib',
          'zmq',
          'dill',
          'mpi4py',
          'cloudpickle>=0.5.5',
          'click',
          'opencv-python',
          'numpy',
          'pandas',
          'matplotlib',
          'seaborn',
          'glob2'
      ] + tf_dependency,
      extras_require={
        'tests': [
            'pytest==3.5.1',
            'pytest-cov'
        ],
        'docs': [
            'sphinx',
            'sphinx-autobuild',
            'sphinx-rtd-theme'
        ]
      },
      description='A fork of OpenAI Baselines, implementations of reinforcement learning algorithms.',
      author='Ashley Hill',
      url='https://github.com/hill-a/stable-baselines',
      author_email='ashley.hill@u-psud.fr',
      keywords="reinforcement-learning-algorithms reinforcement-learning machine-learning "
               "gym openai baselines toolbox python data-science",
      license="MIT",
      long_description=long_description,
      long_description_content_type='text/markdown',
      version="2.5.1a0",
      )

# python setup.py sdist
# python setup.py bdist_wheel
# twine upload --repository-url https://test.pypi.org/legacy/ dist/*
# twine upload dist/*
