.. _quickstart:

===============
Getting Started
===============

Most of the library tries to follow a sklearn-like syntax for the Reinforcement Learning algorithms.

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
`the environment is registered in Gym <https://github.com/openai/gym/wiki/Environments>`_ and if
`the policy is registered <custom_policy.html>`_:

.. code-block:: python

    from stable_baselines import PPO2

    model = PPO2('MlpPolicy', 'CartPole-v1').learn(10000)


.. figure:: https://cdn-images-1.medium.com/max/960/1*R_VMmdgKAY0EDhEjHVelzw.gif

  Define and train a RL agent in one line of code!
