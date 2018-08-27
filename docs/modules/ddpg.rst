.. _ddpg:

.. automodule:: stable_baselines.ddpg


DDPG
====
`Deep Deterministic Policy Gradient (DDPG) <https://arxiv.org/abs/1509.02971>`_


Notes
-----

- Original paper: https://arxiv.org/abs/1509.02971
- Baselines post: https://blog.openai.com/better-exploration-with-parameter-noise/
- ``python -m baselines.ddpg.main`` runs the algorithm for 1M frames = 10M timesteps
  on a Mujoco environment. See help (``-h``) for more options.

Can I use?
----------

-  Reccurent policies: ❌
-  Multi processing: ❌
-  Gym spaces:


============= ====== ===========
Space         Action Observation
============= ====== ===========
Discrete      ❌      ✔️
Box           ✔️      ✔️
MultiDiscrete ❌      ✔️
MultiBinary   ❌      ✔️
============= ====== ===========


Example
-------

.. code-block:: python

  import gym

  from stable_baselines.common.policies import MlpPolicy, CnnPolicy
  from stable_baselines.common.vec_env import DummyVecEnv
  from stable_baselines.ddpg.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise, AdaptiveParamNoiseSpec
  from stable_baselines import DDPG

  env = gym.make('CartPole-v1')
  env = DummyVecEnv([lambda: env])

  # the noise objects for DDPG
  param_noise = None
  action_noise = NormalActionNoise(mean=1, sigma=0)

  model = DDPG(MlpPolicy, env, verbose=1, param_noise=param_noise, action_noise=action_noise)
  model.learn(total_timesteps=25000)
  model.save("ddpg_cartpole")

  del model # remove to demonstrate saving and loading

  DDPG.load("ddpg_cartpole")

  obs = env.reset()
  while True:
      action, _states = model.predict(obs)
      obs, rewards, dones, info = env.step(action)
      env.render()

Parameters
----------

.. autoclass:: DDPG
  :members:
