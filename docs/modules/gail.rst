.. _gail:

.. automodule:: stable_baselines.gail


GAIL
====

The `Generative Adversarial Imitation Learning (GAIL) <https://arxiv.org/abs/1606.03476>`_ uses expert trajectories
to recover a cost function and then learn a policy.

Learning a cost function from expert demonstrations is called Inverse Reinforcement Learning (IRL).
The connection between GAIL and Generative Adversarial Networks (GANs) is that it uses a discriminator that tries
to seperate expert trajectory from trajectories of the learned policy, which has the role of the generator here.


Notes
-----

- Original paper: https://arxiv.org/abs/1606.03476

.. warning::

    Images are not yet handled properly by the current implementation



If you want to train an imitation learning agent
------------------------------------------------


Step 1: Generate expert data
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

You can either train a RL algorithm in a classic setting, use another controller (e.g. a PID controller)
or human demonstrations.

We recommend you to take a look at :ref:`pre-training <pretrain>` section
or directly look at ``stable_baselines/gail/dataset/`` folder to learn more about the expected format for the dataset.

Here is an example of training a Soft Actor-Critic model to generate expert trajectories for GAIL:


.. code-block:: python

  from stable_baselines import SAC
  from stable_baselines.gail import generate_expert_traj

  # Generate expert trajectories (train expert)
  model = SAC('MlpPolicy', 'Pendulum-v0', verbose=1)
  # Train for 60000 timesteps and record 10 trajectories
  # all the data will be saved in 'expert_pendulum.npz' file
  generate_expert_traj(model, 'expert_pendulum', n_timesteps=60000, n_episodes=10)



Step 2: Run GAIL
~~~~~~~~~~~~~~~~


**In case you want to run Behavior Cloning (BC)**

Use the ``.pretrain()`` method (cf guide).


**Others**

Thanks to the open source:

-  @openai/imitation
-  @carpedm20/deep-rl-tensorflow


Can I use?
----------

-  Recurrent policies: ❌
-  Multi processing: ✔️ (using MPI)
-  Gym spaces:


============= ====== ===========
Space         Action Observation
============= ====== ===========
Discrete      ✔️       ✔️
Box           ✔️       ✔️
MultiDiscrete ❌      ✔️
MultiBinary   ❌      ✔️
============= ====== ===========


Example
-------

.. code-block:: python

  import gym

  from stable_baselines import GAIL, SAC
  from stable_baselines.gail import ExpertDataset, generate_expert_traj

  # Generate expert trajectories (train expert)
  model = SAC('MlpPolicy', 'Pendulum-v0', verbose=1)
  generate_expert_traj(model, 'expert_pendulum', n_timesteps=100, n_episodes=10)

  # Load the expert dataset
  dataset = ExpertDataset(expert_path='expert_pendulum.npz', traj_limitation=10, verbose=1)

  model = GAIL("MlpPolicy", 'Pendulum-v0', dataset, verbose=1)
  # Note: in practice, you need to train for 1M steps to have a working policy
  model.learn(total_timesteps=1000)
  model.save("gail_pendulum")

  del model # remove to demonstrate saving and loading

  model = GAIL.load("gail_pendulum")

  env = gym.make('Pendulum-v0')
  obs = env.reset()
  while True:
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    env.render()


Parameters
----------

.. autoclass:: GAIL
  :members:
  :inherited-members:
