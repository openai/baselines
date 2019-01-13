.. _acer:

.. automodule:: stable_baselines.acer


ACER
====

 `Sample Efficient Actor-Critic with Experience Replay (ACER) <https://arxiv.org/abs/1611.01224>`_ combines
 several ideas of previous algorithms: it uses multiple workers (as A2C), implements a replay buffer (as in DQN),
 uses Retrace for Q-value estimation, importance sampling and a trust region.


Notes
-----

- Original paper: https://arxiv.org/abs/1611.01224
- ``python -m stable_baselines.acer.run_atari`` runs the algorithm for 40M frames = 10M timesteps on an Atari game. See help (``-h``) for more options.

Can I use?
----------

-  Recurrent policies: ✔️
-  Multi processing: ✔️
-  Gym spaces:


============= ====== ===========
Space         Action Observation
============= ====== ===========
Discrete      ✔️      ✔️
Box           ❌      ✔️
MultiDiscrete ❌      ✔️
MultiBinary   ❌      ✔️
============= ====== ===========


Example
-------

.. code-block:: python

  import gym

  from stable_baselines.common.policies import MlpPolicy, MlpLstmPolicy, MlpLnLstmPolicy
  from stable_baselines.common.vec_env import SubprocVecEnv
  from stable_baselines import ACER

  # multiprocess environment
  n_cpu = 4
  env = SubprocVecEnv([lambda: gym.make('CartPole-v1') for i in range(n_cpu)])

  model = ACER(MlpPolicy, env, verbose=1)
  model.learn(total_timesteps=25000)
  model.save("acer_cartpole")

  del model # remove to demonstrate saving and loading

  model = ACER.load("acer_cartpole")

  obs = env.reset()
  while True:
      action, _states = model.predict(obs)
      obs, rewards, dones, info = env.step(action)
      env.render()


Parameters
----------

.. autoclass:: ACER
  :members:
  :inherited-members:
