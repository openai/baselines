.. _dqn:

.. automodule:: stable_baselines.deepq


DQN
===

`Deep Q Network (DQN) <https://arxiv.org/abs/1312.5602>`_
and its extensions (Double-DQN, Dueling-DQN, Prioritized Experience Replay).

Notes
-----

- Original paper: https://arxiv.org/abs/1312.5602


Can I use?
----------

-  Reccurent policies: ❌
-  Multi processing: ❌
-  Gym spaces:


============= ====== ===========
Space         Action Observation
============= ====== ===========
Discrete      ✔️      ✔️
Box           ❌       ✔️
MultiDiscrete  ❌       ✔️
MultiBinary    ❌       ✔️
============= ====== ===========


Example
-------

.. code-block:: python
  import gym

  from stable_baselines.common.vec_env import DummyVecEnv
  from stable_baselines.deepq.models import mlp, cnn_to_mlp
  from stable_baselines import DeepQ

  env = gym.make('CartPole-v1')
  env = DummyVecEnv([lambda: env])

  model = DeepQ(mlp(hiddens=[32]), env, verbose=1)
  model.learn(total_timesteps=25000)
  model.save("deepq_cartpole")

  del model # remove to demonstrate saving and loading

  DeepQ.load("deepq_cartpole")

  obs = env.reset()
  while True:
      action, _states = model.predict(obs)
      obs, rewards, dones, info = env.step(action)
      env.render()


With Atari:

.. code-block:: python

  from stable_baselines.common.atari_wrappers import make_atari
  from stable_baselines.deepq.models import mlp, cnn_to_mlp
  from stable_baselines import DeepQ

  env = make_atari('BreakoutNoFrameskip-v4')

  # nature CNN for DeepQ
  cnn_policy = cnn_to_mlp(
  	convs=[(32, 8, 4), (64, 4, 2), (64, 3, 1)],
      hiddens=[256],
      dueling=True)

  model = DeepQ(cnn_policy, env, verbose=1)
  model.learn(total_timesteps=25000)
  model.save("deepq_breakout")

  del model # remove to demonstrate saving and loading

  DeepQ.load("deepq_breakout")

  obs = env.reset()
  while True:
      action, _states = model.predict(obs)
      obs, rewards, dones, info = env.step(action)
      env.render()

Parameters
----------

.. autoclass:: DeepQ
  :members:
