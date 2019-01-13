.. _sac:

.. automodule:: stable_baselines.sac


SAC
===
`Soft Actor Critic (SAC) <https://spinningup.openai.com/en/latest/algorithms/sac.html>`_ Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor.

.. warning::

  The SAC model does not support ``stable_baselines.common.policies`` because it uses double q-values
  and value estimation, as a result it must use its own policy models (see :ref:`sac_policies`).


.. rubric:: Available Policies

.. autosummary::
    :nosignatures:

    MlpPolicy
    LnMlpPolicy
    CnnPolicy
    LnCnnPolicy

Notes
-----

- Original paper: https://arxiv.org/abs/1801.01290
- OpenAI Spinning Guide for SAC: https://spinningup.openai.com/en/latest/algorithms/sac.html
- Original Implementation: https://github.com/haarnoja/sac
- Blog post on using SAC with real robots: https://bair.berkeley.edu/blog/2018/12/14/sac/

.. note::
    In our implementation, we use an entropy coefficient (as in OpenAI Spinning or Facebook Horizon),
    which is the equivalent to the inverse of reward scale in the original SAC paper.
    The main reason is that it avoids having too high errors when updating the Q functions.


.. note::

    The default policies for SAC differ a bit from others MlpPolicy: it uses ReLU instead of tanh activation,
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

  from stable_baselines.sac.policies import MlpPolicy
  from stable_baselines.common.vec_env import DummyVecEnv
  from stable_baselines import SAC

  env = gym.make('Pendulum-v0')
  env = DummyVecEnv([lambda: env])

  model = SAC(MlpPolicy, env, verbose=1)
  model.learn(total_timesteps=50000, log_interval=10)
  model.save("sac_pendulum")

  del model # remove to demonstrate saving and loading

  model = SAC.load("sac_pendulum")

  obs = env.reset()
  while True:
      action, _states = model.predict(obs)
      obs, rewards, dones, info = env.step(action)
      env.render()

Parameters
----------

.. autoclass:: SAC
  :members:
  :inherited-members:

.. _sac_policies:

SAC Policies
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

  from stable_baselines.sac.policies import FeedForwardPolicy
  from stable_baselines.common.vec_env import DummyVecEnv
  from stable_baselines import SAC

  # Custom MLP policy of three layers of size 128 each
  class CustomSACPolicy(FeedForwardPolicy):
      def __init__(self, *args, **kwargs):
          super(CustomPolicy, self).__init__(*args, **kwargs,
                                             layers=[128, 128, 128],
                                             layer_norm=False,
                                             feature_extraction="mlp")

  # Create and wrap the environment
  env = gym.make('Pendulum-v0')
  env = DummyVecEnv([lambda: env])

  model = SAC(CustomSACPolicy, env, verbose=1)
  # Train the agent
  model.learn(total_timesteps=100000)
