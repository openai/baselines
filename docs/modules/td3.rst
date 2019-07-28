.. _td3:

.. automodule:: stable_baselines.td3


TD3
===

`Twin Delayed DDPG (TD3) <https://spinningup.openai.com/en/latest/algorithms/td3.html>`_ Addressing Function Approximation Error in Actor-Critic Methods.

TD3 is a direct successor of DDPG and improves it using three major tricks: clipped double Q-Learning, delayed policy update and target policy smoothing.
We recommend reading `OpenAI Spinning guide on TD3 <https://spinningup.openai.com/en/latest/algorithms/td3.html>`_ to learn more about those.


.. warning::

  The TD3 model does not support ``stable_baselines.common.policies`` because it uses double q-values
  estimation, as a result it must use its own policy models (see :ref:`td3_policies`).


.. rubric:: Available Policies

.. autosummary::
    :nosignatures:

    MlpPolicy
    LnMlpPolicy
    CnnPolicy
    LnCnnPolicy

Notes
-----

- Original paper: https://arxiv.org/pdf/1802.09477.pdf
- OpenAI Spinning Guide for TD3: https://spinningup.openai.com/en/latest/algorithms/td3.html
- Original Implementation: https://github.com/sfujim/TD3

.. note::

    The default policies for TD3 differ a bit from others MlpPolicy: it uses ReLU instead of tanh activation,
    to match the original paper


Can I use?
----------

-  Recurrent policies: ❌
-  Multi processing: ❌
-  Gym spaces:


============= ====== ===========
Space         Action Observation
============= ====== ===========
Discrete      ❌      ✔️
Box           ✔️       ✔️
MultiDiscrete ❌      ✔️
MultiBinary   ❌      ✔️
============= ====== ===========


Example
-------

.. code-block:: python

  import gym
  import numpy as np

  from stable_baselines import TD3
  from stable_baselines.td3.policies import MlpPolicy
  from stable_baselines.common.vec_env import DummyVecEnv
  from stable_baselines.ddpg.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise

  env = gym.make('Pendulum-v0')
  env = DummyVecEnv([lambda: env])

  # The noise objects for TD3
  n_actions = env.action_space.shape[-1]
  action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))

  model = TD3(MlpPolicy, env, action_noise=action_noise, verbose=1)
  model.learn(total_timesteps=50000, log_interval=10)
  model.save("td3_pendulum")

  del model # remove to demonstrate saving and loading

  model = TD3.load("td3_pendulum")

  obs = env.reset()
  while True:
      action, _states = model.predict(obs)
      obs, rewards, dones, info = env.step(action)
      env.render()

Parameters
----------

.. autoclass:: TD3
  :members:
  :inherited-members:

.. _td3_policies:

TD3 Policies
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


Custom Policy Network
---------------------

Similarly to the example given in the `examples <../guide/custom_policy.html>`_ page.
You can easily define a custom architecture for the policy network:

.. code-block:: python

  import gym
  import numpy as np

  from stable_baselines import TD3
  from stable_baselines.td3.policies import FeedForwardPolicy
  from stable_baselines.common.vec_env import DummyVecEnv
  from stable_baselines.ddpg.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise

  # Custom MLP policy with two layers
  class CustomTD3Policy(FeedForwardPolicy):
      def __init__(self, *args, **kwargs):
          super(CustomTD3Policy, self).__init__(*args, **kwargs,
                                             layers=[400, 300],
                                             layer_norm=False,
                                             feature_extraction="mlp")

  # Create and wrap the environment
  env = gym.make('Pendulum-v0')
  env = DummyVecEnv([lambda: env])

  # The noise objects for TD3
  n_actions = env.action_space.shape[-1]
  action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))


  model = TD3(CustomTD3Policy, env, action_noise=action_noise, verbose=1)
  # Train the agent
  model.learn(total_timesteps=80000)
