.. _tensorboard:

Using Tensorboard
==========================

To use the Tensorboard with the rl baselines, you simply need to define a log location for the RL agent:

.. code-block:: python

    import gym

    from stable_baselines.common.policies import MlpPolicy
    from stable_baselines.common.vec_env import DummyVecEnv
    from stable_baselines import A2C

    env = gym.make('CartPole-v1')
    env = DummyVecEnv([lambda: env])  # The algorithms require a vectorized environment to run

    model = A2C(MlpPolicy, env, verbose=1, tensorboard_log="./a2c_cartpole_tensorboard/")
    model.learn(total_timesteps=10000)


Or after loading an existing model (by default the log path is not saved):

.. code-block:: python

    import gym

    from stable_baselines.common.vec_env import DummyVecEnv
    from stable_baselines import A2C

    env = gym.make('CartPole-v1')
    env = DummyVecEnv([lambda: env])  # The algorithms require a vectorized environment to run

    model = A2C.load("./a2c_cartpole.pkl", env=env, tensorboard_log="./a2c_cartpole_tensorboard/")
    model.learn(total_timesteps=10000)


Then you can monitor the RL agent during or after the training, with the following bash command:

.. code-block:: bash

  tensorboard --logdir ./a2c_cartpole/

It will display information such as the model graph, the episode reward, the model losses, the observation and other parameter unique to some models.

image:: ../static/img/Tensorboard_example_1.png

  :width: 400
  :alt: plotting

image:: ../static/img/Tensorboard_example_1.png

  :width: 400
  :alt: histogram

image:: ../static/img/Tensorboard_example_1.png

  :width: 400
  :alt: graph
