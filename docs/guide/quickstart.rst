.. _quickstart:

===============
Getting Started
===============

Most of the library tries to follow a sklearn-like syntax for the Reinforcement Learning algorithms.

Basic Examples
~~~~~~~~~~~~~~

Here is a quick example of how to train and run PPO2 on a cartpole environment:

.. code-block:: python

  import gym

  from stable_baselines.common.policies import MlpPolicy
  from stable_baselines.common.vec_env import DummyVecEnv
  from stable_baselines import PPO2

  env = gym.make('CartPole-v1')
  env = DummyVecEnv([lambda: env])  # The algorithms require a vectorized environment to run

  model = PPO2(MlpPolicy, env, verbose=1)
  model.learn(total_timesteps=10000)

  obs = env.reset()
  for i in range(1000):
      action, _states = model.predict(obs)
      obs, rewards, dones, info = env.step(action)
      env.render()


Or just train a model with a one liner if
`the environment is registered in Gym <https://github.com/openai/gym/wiki/Environments>`_:

.. code-block:: python

    from stable_baselines.common.policies import MlpPolicy
    from stable_baselines import PPO2

    model = PPO2(MlpPolicy, 'CartPole-v1').learn(10000)


Try it online with Colab Notebooks!
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

All the following examples can be executed online using Google colab
notebooks:

-  `Getting Started`_
-  `Training, Saving, Loading`_
-  `Multiprocessing`_
-  `Monitor Training and Plotting`_
-  `Atari Games`_
-  `Breakout`_ (trained agent included)

.. _Getting Started: https://colab.research.google.com/drive/1_1H5bjWKYBVKbbs-Kj83dsfuZieDNcFU
.. _Training, Saving, Loading: https://colab.research.google.com/drive/1KoAQ1C_BNtGV3sVvZCnNZaER9rstmy0s
.. _Multiprocessing: https://colab.research.google.com/drive/1ZzNFMUUi923foaVsYb4YjPy4mjKtnOxb
.. _Monitor Training and Plotting: https://colab.research.google.com/drive/1L_IMo6v0a0ALK8nefZm6PqPSy0vZIWBT
.. _Atari Games: https://colab.research.google.com/drive/1iYK11yDzOOqnrXi1Sfjm1iekZr4cxLaN
.. _Breakout: https://colab.research.google.com/drive/14NwwEHwN4hdNgGzzySjxQhEVDff-zr7O
