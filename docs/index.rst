.. Stable Baselines documentation master file, created by
   sphinx-quickstart on Sat Aug 25 10:33:54 2018.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to Stable Baselines docs!
============================================

`Stable Baselines <https://github.com/hill-a/stable-baselines>`_ is a set of improved implementations
of reinforcement learning algorithms based on OpenAI `Baselines <https://github.com/openai/baselines>`_.


Train a RL agent in one line of code!

.. code-block:: python

  from stable_baselines.common.policies import MlpPolicy
  from stable_baselines import PPO2

  model = PPO2(MlpPolicy, 'CartPole-v1').learn(10000)

Table of Contents
=================
.. toctree::
   :maxdepth: 2
   :numbered:

   install
   quickstart



Indices and tables
==================

* :ref:`genindex`
.. * :ref:`modindex`
* :ref:`search`
