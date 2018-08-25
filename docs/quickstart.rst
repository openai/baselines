.. _quickstart:

===============
Getting Started
===============

.. note::

    note

.. code-block:: python

  from stable_baselines.common.policies import MlpPolicy
  from stable_baselines import PPO2

  model = PPO2(MlpPolicy, 'CartPole-v1').learn(10000)
