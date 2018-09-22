.. _ppo2:

.. automodule:: stable_baselines.ppo2

PPO2
====

The `Proximal Policy Optimization <https://arxiv.org/abs/1707.06347>`_ algorithm combines ideas from A2C (having multiple workers)
and TRPO (it uses a trust region to improve the actor).

The main idea is that after an update, the new policy should be not too far form the `old` policy.
For that, ppo uses clipping to avoid too large update.

.. note::

  PPO2 is the implementation of OpenAI made for GPU. For multiprocessing, it uses vectorized environments
  compared to PPO1 which uses MPI.

.. note::

  PPO2 contains several modifications from the original algorithm not documented
  by OpenAI: value function is also clipped and advantages are normalized.


Notes
-----

-  Original paper: https://arxiv.org/abs/1707.06347
-  OpenAI blog post: https://blog.openai.com/openai-baselines-ppo/
-  ``python -m stable_baselines.ppo2.run_atari`` runs the algorithm for 40M
   frames = 10M timesteps on an Atari game. See help (``-h``) for more
   options.
-  ``python -m stable_baselines.ppo2.run_mujoco`` runs the algorithm for 1M
   frames on a Mujoco environment.

Can I use?
----------

-  Reccurent policies: ✔️
-  Multi processing: ✔️
-  Gym spaces:


============= ====== ===========
Space         Action Observation
============= ====== ===========
Discrete      ✔️      ✔️
Box           ✔️      ✔️
MultiDiscrete ✔️      ✔️
MultiBinary   ✔️      ✔️
============= ====== ===========

Example
-------

Train a PPO agent on `CartPole-v1` using 4 processes.

.. code-block:: python

   import gym

   from stable_baselines.common.policies import MlpPolicy
   from stable_baselines.common.vec_env import SubprocVecEnv
   from stable_baselines import PPO2

   # multiprocess environment
   n_cpu = 4
   env = SubprocVecEnv([lambda: gym.make('CartPole-v1') for i in range(n_cpu)])

   model = PPO2(MlpPolicy, env, verbose=1)
   model.learn(total_timesteps=25000)
   model.save("ppo2_cartpole")

   del model # remove to demonstrate saving and loading

   model = PPO2.load("ppo2_cartpole")

   # Enjoy trained agent
   obs = env.reset()
   while True:
       action, _states = model.predict(obs)
       obs, rewards, dones, info = env.step(action)
       env.render()

Parameters
----------

.. autoclass:: PPO2
  :members:
  :inherited-members:
