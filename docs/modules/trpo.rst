.. _trpo:

.. automodule:: stable_baselines.trpo_mpi


TRPO
====

`Trust Region Policy Optimization (TRPO) <https://arxiv.org/abs/1502.05477>`_
is an iterative approach for optimizing policies with guaranteed monotonic improvement.

Notes
-----

-  Original paper:  https://arxiv.org/abs/1502.05477
-  OpenAI blog post: https://blog.openai.com/openai-baselines-ppo/
- ``mpirun -np 16 python -m stable_baselines.trpo_mpi.run_atari`` runs the algorithm
  for 40M frames = 10M timesteps on an Atari game. See help (``-h``) for more options.
- ``python -m stable_baselines.trpo_mpi.run_mujoco`` runs the algorithm for 1M timesteps on a Mujoco environment.

Can I use?
----------

-  Reccurent policies: ✔️
-  Multi processing: ✔️  (using MPI)
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

  from stable_baselines.common.policies import MlpPolicy, MlpLstmPolicy, MlpLnLstmPolicy, \
      CnnPolicy, CnnLstmPolicy, CnnLnLstmPolicy
  from stable_baselines.common.vec_env import DummyVecEnv
  from stable_baselines import TRPO

  env = gym.make('CartPole-v1')
  env = DummyVecEnv([lambda: env])

  model = TRPO(MlpPolicy, env, verbose=1)
  model.learn(total_timesteps=25000)
  model.save("trpo_cartpole")

  del model # remove to demonstrate saving and loading

  model = TRPO.load("trpo_cartpole")

  obs = env.reset()
  while True:
      action, _states = model.predict(obs)
      obs, rewards, dones, info = env.step(action)
      env.render()


Parameters
----------

.. autoclass:: TRPO
  :members:
  :inherited-members:
