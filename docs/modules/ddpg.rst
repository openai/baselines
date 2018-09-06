.. _ddpg:

.. automodule:: stable_baselines.ddpg


DDPG
====
`Deep Deterministic Policy Gradient (DDPG) <https://arxiv.org/abs/1509.02971>`_


.. warning::

  The DDPG model does not support Actor critic policies,
  as a result it must use its own policy models (see :ref:`ddpg_policies`).

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
  import numpy as np

  from stable_baselines.ddpg.policies import MlpPolicy, CnnPolicy
  from stable_baselines.common.vec_env import DummyVecEnv
  from stable_baselines.ddpg.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise, AdaptiveParamNoiseSpec
  from stable_baselines import DDPG

  env = gym.make('MountainCarContinuous-v0')
  env = DummyVecEnv([lambda: env])

  # the noise objects for DDPG
  n_actions = env.action_space.shape[-1]
  param_noise = None
  action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=float(0.2) * np.ones(n_actions))

  model = DDPG(MlpPolicy, env, verbose=1, param_noise=param_noise, action_noise=action_noise)
  model.learn(total_timesteps=25000)
  model.save("ddpg_mountain")

  del model # remove to demonstrate saving and loading

  DDPG.load("ddpg_mountain")

  obs = env.reset()
  while True:
      action, _states = model.predict(obs)
      obs, rewards, dones, info = env.step(action)
      env.render()

Parameters
----------

.. autoclass:: DDPG
  :members:
  :inherited-members:

.. _ddpg_policies:

DDPG Policies
-------------

.. autoclass:: MlpPolicy
  :members:
  :inherited-members:


.. autoclass:: CnnPolicy
  :members:
  :inherited-members:

Action and Parameters Noise
---------------------------

.. autoclass:: AdaptiveParamNoiseSpec
  :members:
  :inherited-members:

.. autoclass:: NormalActionNoise
  :members:
  :inherited-members:

.. autoclass:: OrnsteinUhlenbeckActionNoise
  :members:
  :inherited-members:
