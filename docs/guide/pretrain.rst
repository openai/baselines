.. _pretrain:

.. automodule:: stable_baselines.gail


Pre-Training (Behavior Cloning)
===============================

With the ``.pretrain()`` method, you can pre-train RL policies using trajectories from an expert, and therefore accelerate training.

Behavior Cloning (BC) treats the problem of imitation learning, i.e., using expert demonstrations, as a supervised learning problem.
That is to say, given expert trajectories (observations-actions pairs), the policy network is trained to reproduce the expert behavior:
for a given observation, the action taken by the policy must be the one taken by the expert.

Expert trajectories can be human demonstrations, trajectories from another controller (e.g. a PID controller)
or trajectories from a trained RL agent.


.. note::

	Only ``Box`` and ``Discrete`` spaces are supported for now for pre-training a model.


.. note::

  Images datasets are treated a bit differently as other datasets to avoid memory issues.
  The images from the expert demonstrations must be located in a folder, not in the expert numpy archive.



Generate Expert Trajectories
----------------------------

Here, we are going to train a RL model and then generate expert trajectories
using this agent.

Note that in practice, generating expert trajectories usually does not require training an RL agent.

The following example is only meant to demonstrate the ``pretrain()`` feature.

However, we recommend users to take a look at the code of the ``generate_expert_traj()`` function (located in ``gail/dataset/`` folder)
to learn about the data structure of the expert dataset (see below for an overview) and how to record trajectories.


.. code-block:: python

  from stable_baselines import DQN
  from stable_baselines.gail import generate_expert_traj

  model = DQN('MlpPolicy', 'CartPole-v1', verbose=1)
	# Train a DQN agent for 1e5 timesteps and generate 10 trajectories
	# data will be saved in a numpy archive named `expert_cartpole.npz`
  generate_expert_traj(model, 'expert_cartpole', n_timesteps=int(1e5), n_episodes=10)



Here is an additional example when the expert controller is a callable,
that is passed to the function instead of a RL model.
The idea is that this callable can be a PID controller, asking a human player, ...


.. code-block:: python

		import gym

		from stable_baselines.gail import generate_expert_traj

		env = gym.make("CartPole-v1")
		# Here the expert is a random agent
		# but it can be any python function, e.g. a PID controller
		def dummy_expert(_obs):
		    """
		    Random agent. It samples actions randomly
		    from the action space of the environment.

		    :param _obs: (np.ndarray) Current observation
		    :return: (np.ndarray) action taken by the expert
		    """
		    return env.action_space.sample()
		# Data will be saved in a numpy archive named `expert_cartpole.npz`
		# when using something different than an RL expert,
		# you must pass the environment object explicitely
		generate_expert_traj(dummy_expert, 'dummy_expert_cartpole', env, n_episodes=10)



Pre-Train a Model using Behavior Cloning
----------------------------------------

Using the ``expert_cartpole.npz`` dataset generated with the previous script.

.. code-block:: python

	from stable_baselines import PPO2
	from stable_baselines.gail import ExpertDataset
	# Using only one expert trajectory
	# you can specify `traj_limitation=-1` for using the whole dataset
	dataset = ExpertDataset(expert_path='expert_cartpole.npz',
	                        traj_limitation=1, batch_size=128)

	model = PPO2('MlpPolicy', 'CartPole-v1', verbose=1)
	# Pretrain the PPO2 model
	model.pretrain(dataset, n_epochs=1000)

	# As an option, you can train the RL agent
	# model.learn(int(1e5))

	# Test the pre-trained model
	env = model.get_env()
	obs = env.reset()

	reward_sum = 0.0
	for _ in range(1000):
		action, _ = model.predict(obs)
		obs, reward, done, _ = env.step(action)
		reward_sum += reward
		env.render()
		if done:
			print(reward_sum)
			reward_sum = 0.0
			obs = env.reset()

	env.close()


Data Structure of the Expert Dataset
------------------------------------

The expert dataset is a ``.npz`` archive. The data is saved in python dictionary format with keys: ``actions``, ``episode_returns``, ``rewards``, ``obs``,
``episode_starts``.

In case of images, ``obs`` contains the relative path to the images.

obs, actions: shape (N * L, ) + S

where N = # episodes, L = episode length
and S is the environment observation/action space.

S = (1, ) for discrete space


.. autoclass:: ExpertDataset
  :members:
  :inherited-members:


.. autoclass:: DataLoader
  :members:
  :inherited-members:


.. autofunction:: generate_expert_traj
