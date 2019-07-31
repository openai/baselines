.. _tensorboard:

Tensorboard Integration
==========================

Basic Usage
------------

To use Tensorboard with the rl baselines, you simply need to define a log location for the RL agent:

.. code-block:: python

    import gym

    from stable_baselines import A2C

    model = A2C('MlpPolicy', 'CartPole-v1', verbose=1, tensorboard_log="./a2c_cartpole_tensorboard/")
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


You can also define custom logging name when training (by default it is the algorithm name)

.. code-block:: python

    import gym

    from stable_baselines import A2C

    model = A2C('MlpPolicy', 'CartPole-v1', verbose=1, tensorboard_log="./a2c_cartpole_tensorboard/")
    model.learn(total_timesteps=10000, tb_log_name="first_run")
    # Pass reset_num_timesteps=False to continue the training curve in tensorboard
    # By default, it will create a new curve
    model.learn(total_timesteps=10000, tb_log_name="second_run", reset_num_timesteps=False)
    model.learn(total_timesteps=10000, tb_log_name="thrid_run", reset_num_timesteps=False)


Once the learn function is called, you can monitor the RL agent during or after the training, with the following bash command:

.. code-block:: bash

  tensorboard --logdir ./a2c_cartpole_tensorboard/

you can also add past logging folders:

.. code-block:: bash

  tensorboard --logdir ./a2c_cartpole_tensorboard/;./ppo2_cartpole_tensorboard/

It will display information such as the model graph, the episode reward, the model losses, the observation and other parameter unique to some models.

.. image:: ../_static/img/Tensorboard_example_1.png
  :width: 400
  :alt: plotting

.. image:: ../_static/img/Tensorboard_example_2.png
  :width: 400
  :alt: histogram

.. image:: ../_static/img/Tensorboard_example_3.png
  :width: 400
  :alt: graph


Logging More Values
-------------------

Using a callback, you can easily log more values with TensorBoard.
Here is a simple example on how to log both additional tensor or arbitrary scalar value:

.. code-block:: python

  import tensorflow as tf
  import numpy as np

  from stable_baselines import SAC

  model = SAC("MlpPolicy", "Pendulum-v0", tensorboard_log="/tmp/sac/", verbose=1)
  # Define a new property to avoid global variable
  model.is_tb_set = False


  def callback(locals_, globals_):
      self_ = locals_['self']
      # Log additional tensor
      if not self_.is_tb_set:
          with self_.graph.as_default():
              tf.summary.scalar('value_target', tf.reduce_mean(self_.value_target))
              self_.summary = tf.summary.merge_all()
          self_.is_tb_set = True
      # Log scalar value (here a random variable)
      value = np.random.random()
      summary = tf.Summary(value=[tf.Summary.Value(tag='random_value', simple_value=value)])
      locals_['writer'].add_summary(summary, self_.num_timesteps)
      return True


  model.learn(50000, callback=callback)

Legacy Integration
-------------------

All the information displayed in the terminal (default logging) can be also logged in tensorboard.
For that, you need to define several environment variables:

.. code-block:: bash

  # formats are comma-separated, but for tensorboard you only need the last one
  # stdout -> terminal
  export OPENAI_LOG_FORMAT='stdout,log,csv,tensorboard'
  export OPENAI_LOGDIR=path/to/tensorboard/data

and to configure the logger using:

.. code-block:: python

  from stable_baselines.logger import configure

  configure()


Then start tensorboard with:

.. code-block:: bash

  tensorboard --logdir=$OPENAI_LOGDIR
