.. _rl_zoo:

=================
RL Baselines Zoo
=================

`RL Baselines Zoo <https://github.com/araffin/rl-baselines-zoo>`_. is a collection of pre-trained Reinforcement Learning agents using
Stable-Baselines.
It also provides basic scripts for training, evaluating agents and recording videos.

Goals of this repository:

1. Provide a simple interface to train and enjoy RL agents
2. Benchmark the different Reinforcement Learning algorithms
3. Provide tuned hyperparameters for each environment and RL algorithm
4. Have fun with the trained agents!

Installation
------------

1. Install dependencies
::

   apt-get install swig cmake libopenmpi-dev zlib1g-dev ffmpeg
   pip install stable-baselines>=2.2.1 box2d box2d-kengz pyyaml pybullet==2.1.0 pytablewriter

2. Clone the repository:

::

  git clone https://github.com/araffin/rl-baselines-zoo


Train an Agent
--------------

The hyperparameters for each environment are defined in
``hyperparameters/algo_name.yml``.

If the environment exists in this file, then you can train an agent
using:

::

 python train.py --algo algo_name --env env_id

For example (with tensorboard support):

::

 python train.py --algo ppo2 --env CartPole-v1 --tensorboard-log /tmp/stable-baselines/

Train for multiple environments (with one call) and with tensorboard
logging:

::

 python train.py --algo a2c --env MountainCar-v0 CartPole-v1 --tensorboard-log /tmp/stable-baselines/

Continue training (here, load pretrained agent for Breakout and continue
training for 5000 steps):

::

 python train.py --algo a2c --env BreakoutNoFrameskip-v4 -i trained_agents/a2c/BreakoutNoFrameskip-v4.pkl -n 5000


Enjoy a Trained Agent
---------------------

If the trained agent exists, then you can see it in action using:

::

  python enjoy.py --algo algo_name --env env_id

For example, enjoy A2C on Breakout during 5000 timesteps:

::

  python enjoy.py --algo a2c --env BreakoutNoFrameskip-v4 --folder trained_agents/ -n 5000


.. note::

	You can find more information about the rl baselines zoo in the repo `README <https://github.com/araffin/rl-baselines-zoo>`_. For instance, how to record a video of a trained agent.
