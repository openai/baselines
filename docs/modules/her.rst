.. _her:

.. automodule:: stable_baselines.her


HER
====

`Hindsight Experience Replay (HER) <https://arxiv.org/abs/1707.01495>`_

HER is a method wrapper that works with Off policy methods (DQN, SAC, TD3 and DDPG for example).

.. note::

	HER was re-implemented from scratch in Stable-Baselines compared to the original OpenAI baselines.
	If you want to reproduce results from the paper, please use the rl baselines zoo
	in order to have the correct hyperparameters and at least 8 MPI workers with DDPG.

.. warning::

	HER requires the environment to inherits from `gym.GoalEnv <https://github.com/openai/gym/blob/3394e245727c1ae6851b504a50ba77c73cd4c65b/gym/core.py#L160>`_


.. warning::

	you must pass an environment or wrap it with ``HERGoalEnvWrapper`` in order to use the predict method


Notes
-----

- Original paper: https://arxiv.org/abs/1707.01495
- OpenAI paper: `Plappert et al. (2018)`_
- OpenAI blog post: https://openai.com/blog/ingredients-for-robotics-research/


.. _Plappert et al. (2018): https://arxiv.org/abs/1802.09464

Can I use?
----------

Please refer to the wrapped model (DQN, SAC, TD3 or DDPG) for that section.

Example
-------

.. code-block:: python

	from stable_baselines import HER, DQN, SAC, DDPG, TD3
	from stable_baselines.her import GoalSelectionStrategy, HERGoalEnvWrapper
	from stable_baselines.common.bit_flipping_env import BitFlippingEnv

	model_class = DQN  # works also with SAC, DDPG and TD3

	env = BitFlippingEnv(N_BITS, continuous=model_class in [DDPG, SAC, TD3], max_steps=N_BITS)

	# Available strategies (cf paper): future, final, episode, random
	goal_selection_strategy = 'future' # equivalent to GoalSelectionStrategy.FUTURE

	# Wrap the model
	model = HER('MlpPolicy', env, model_class, n_sampled_goal=4, goal_selection_strategy=goal_selection_strategy,
							verbose=1)
	# Train the model
	model.learn(1000)

	model.save("./her_bit_env")

	# WARNING: you must pass an env
	# or wrap your environment with HERGoalEnvWrapper to use the predict method
	model = HER.load('./her_bit_env', env=env)

	obs = env.reset()
	for _ in range(100):
	    action, _ = model.predict(obs)
	    obs, reward, done, _ = env.step(action)

	    if done:
	        obs = env.reset()


Parameters
----------

.. autoclass:: HER
  :members:

Goal Selection Strategies
-------------------------

.. autoclass:: GoalSelectionStrategy
  :members:
  :inherited-members:
	:undoc-members:


Gaol Env Wrapper
----------------

.. autoclass:: HERGoalEnvWrapper
  :members:
  :inherited-members:
	:undoc-members:


Replay Wrapper
--------------

.. autoclass:: HindsightExperienceReplayWrapper
  :members:
  :inherited-members:
