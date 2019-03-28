.. _ppo1:

.. automodule:: stable_baselines.ppo1


PPO1
====

The `Proximal Policy Optimization <https://arxiv.org/abs/1707.06347>`_ algorithm combines ideas from A2C (having multiple workers)
and TRPO (it uses a trust region to improve the actor).

The main idea is that after an update, the new policy should be not too far form the `old` policy.
For that, ppo uses clipping to avoid too large update.

.. note::

  PPO2 is the implementation of OpenAI made for GPU. For multiprocessing, it uses vectorized environments
  compared to PPO1 which uses MPI.

Notes
-----

-  Original paper:  https://arxiv.org/abs/1707.06347
- Clear explanation of PPO on Arxiv Insights channel: https://www.youtube.com/watch?v=5P7I-xPq8u8
-  OpenAI blog post: https://blog.openai.com/openai-baselines-ppo/
- ``mpirun -np 8 python -m stable_baselines.ppo1.run_atari`` runs the algorithm for 40M frames = 10M timesteps on an Atari game. See help (``-h``) for more options.
- ``python -m stable_baselines.ppo1.run_mujoco`` runs the algorithm for 1M frames on a Mujoco environment.
- Train mujoco 3d humanoid (with optimal-ish hyperparameters): ``mpirun -np 16 python -m stable_baselines.ppo1.run_humanoid --model-path=/path/to/model``
- Render the 3d humanoid: ``python -m stable_baselines.ppo1.run_humanoid --play --model-path=/path/to/model``

Can I use?
----------

-  Recurrent policies: ❌
-  Multi processing: ✔️ (using MPI)
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

.. code-block:: python

  import gym

  from stable_baselines.common.policies import MlpPolicy
  from stable_baselines.common.vec_env import DummyVecEnv
  from stable_baselines import PPO1

  env = gym.make('CartPole-v1')
  env = DummyVecEnv([lambda: env])

  model = PPO1(MlpPolicy, env, verbose=1)
  model.learn(total_timesteps=25000)
  model.save("ppo1_cartpole")

  del model # remove to demonstrate saving and loading

  model = PPO1.load("ppo1_cartpole")

  obs = env.reset()
  while True:
      action, _states = model.predict(obs)
      obs, rewards, dones, info = env.step(action)
      env.render()


Parameters
----------

.. autoclass:: PPO1
  :members:
  :inherited-members:
