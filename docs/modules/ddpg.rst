.. _ddpg:

.. automodule:: stable_baselines.ddpg


DDPG
====
`Deep Deterministic Policy Gradient (DDPG) <https://arxiv.org/abs/1509.02971>`_


.. warning::

  The DDPG model does not support ``stable_baselines.common.policies`` because it uses q-value instead
  of value estimation, as a result it must use its own policy models (see :ref:`ddpg_policies`).


.. rubric:: Available Policies

.. autosummary::
    :nosignatures:

    MlpPolicy
    LnMlpPolicy
    CnnPolicy
    LnCnnPolicy

Notes
-----

- Original paper: https://arxiv.org/abs/1509.02971
- Baselines post: https://blog.openai.com/better-exploration-with-parameter-noise/
- ``python -m stable_baselines.ddpg.main`` runs the algorithm for 1M frames = 10M timesteps
  on a Mujoco environment. See help (``-h``) for more options.

Can I use?
----------

-  Recurrent policies: ❌
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

  from stable_baselines.ddpg.policies import MlpPolicy
  from stable_baselines.common.vec_env import DummyVecEnv
  from stable_baselines.ddpg.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise, AdaptiveParamNoiseSpec
  from stable_baselines import DDPG

  env = gym.make('MountainCarContinuous-v0')
  env = DummyVecEnv([lambda: env])

  # the noise objects for DDPG
  n_actions = env.action_space.shape[-1]
  param_noise = None
  action_noise = OrnsteinUhlenbeckActionNoise(mean=np.zeros(n_actions), sigma=float(0.5) * np.ones(n_actions))

  model = DDPG(MlpPolicy, env, verbose=1, param_noise=param_noise, action_noise=action_noise)
  model.learn(total_timesteps=400000)
  model.save("ddpg_mountain")

  del model # remove to demonstrate saving and loading

  model = DDPG.load("ddpg_mountain")

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


.. autoclass:: LnMlpPolicy
  :members:
  :inherited-members:


.. autoclass:: CnnPolicy
  :members:
  :inherited-members:


.. autoclass:: LnCnnPolicy
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


Custom Policy Network
---------------------

Similarly to the example given in the `examples <../guide/custom_policy.html>`_ page.
You can easily define a custom architecture for the policy network:

.. code-block:: python

  import gym

  from stable_baselines.ddpg.policies import FeedForwardPolicy
  from stable_baselines.common.vec_env import DummyVecEnv
  from stable_baselines import DDPG

  # Custom MLP policy of two layers of size 16 each
  class CustomPolicy(FeedForwardPolicy):
      def __init__(self, *args, **kwargs):
          super(CustomPolicy, self).__init__(*args, **kwargs,
                                             layers=[16, 16],
                                             layer_norm=False,
                                             feature_extraction="mlp")

  # Create and wrap the environment
  env = gym.make('Pendulum-v0')
  env = DummyVecEnv([lambda: env])

  model = DDPG(CustomPolicy, env, verbose=1)
  # Train the agent
  model.learn(total_timesteps=100000)
