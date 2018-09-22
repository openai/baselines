.. _changelog:

Changelog
==========

For download links, please look at `Github release page <https://github.com/hill-a/stable-baselines/releases>`_.

Pre Release 2.0.1.a0 (WIP)
---------------------------

**logging and bug fixes**

- added patch fix for equal function using `gym.spaces.MultiDiscrete` and `gym.spaces.MultiBinary`


Release 2.0.0 (2018-09-18)
--------------------------

.. warning::

	This version contains breaking changes, please read the full details

**Tensorboard, refactoring and bug fixes**


- Renamed DeepQ to DQN **breaking changes**
- Renamed DeepQPolicy to DQNPolicy **breaking changes**
- fixed DDPG behavior **breaking changes**
- changed default policies for DDPG, so that DDPG now works correctly **breaking changes**
- added more documentation (some modules from common).
- added doc about using custom env
- added Tensorboard support for A2C, ACER, ACKTR, DDPG, DeepQ, PPO1, PPO2 and TRPO
- added episode reward to Tensorboard
- added documentation for Tensorboard usage
- added Identity for Box action space
- fixed render function ignoring parameters when using wrapped environments
- fixed PPO1 and TRPO done values for recurrent policies
- fixed image normalization not occurring when using images
- updated VecEnv objects for the new Gym version
- added test for DDPG
- refactored DQN policies
- added registry for policies, can be passed as string to the agent
- added documentation for custom policies + policy registration
- fixed numpy warning when using DDPG Memory
- fixed DummyVecEnv not copying the observation array when stepping and resetting
- added pre-built docker images + installation instructions
- added ``deterministic`` argument in the predict function
- added assert in PPO2 for recurrent policies
- fixed predict function to handle both vectorized and unwrapped environment
- added input check to the predict function
- refactored ActorCritic models to reduce code duplication
- refactored Off Policy models (to begin HER and replay_buffer refactoring)
- added tests for auto vectorization detection
- fixed render function, to handle positional arguments


Release 1.0.7 (2018-08-29)
--------------------------

**Bug fixes and documentation**

- added html documentation using sphinx + integration with read the docs
- cleaned up README + typos
- fixed normalization for DQN with images
- fixed DQN identity test


Release 1.0.1 (2018-08-20)
--------------------------

**Refactored Stable Baselines**

- refactored A2C, ACER, ACTKR, DDPG, DeepQ, GAIL, TRPO, PPO1 and PPO2 under a single constant class
- added callback to refactored algorithm training
- added saving and loading to refactored algorithms
- refactored ACER, DDPG, GAIL, PPO1 and TRPO to fit with A2C, PPO2 and ACKTR policies
- added new policies for most algorithms (Mlp, MlpLstm, MlpLnLstm, Cnn, CnnLstm and CnnLnLstm)
- added dynamic environment switching (so continual RL learning is now feasible)
- added prediction from observation and action probability from observation for all the algorithms
- fixed graphs issues, so models wont collide in names
- fixed behavior_clone weight loading for GAIL
- fixed Tensorflow using all the GPU VRAM
- fixed models so that they are all compatible with vectorized environments
- fixed ```set_global_seed``` to update ```gym.spaces```'s random seed
- fixed PPO1 and TRPO performance issues when learning identity function
- added new tests for loading, saving, continuous actions and learning the identity function
- fixed DQN wrapping for atari
- added saving and loading for Vecnormalize wrapper
- added automatic detection of action space (for the policy network)
- fixed ACER buffer with constant values assuming n_stack=4
- fixed some RL algorithms not clipping the action to be in the action_space, when using ```gym.spaces.Box```
- refactored algorithms can take either a ```gym.Environment``` or a ```str``` ([if the environment name is registered](https://github.com/openai/gym/wiki/Environments))
- Hoftix in ACER (compared to v1.0.0)

Future Work :

- Finish refactoring HER
- Refactor ACKTR and ACER for continuous implementation



Release 0.1.6 (2018-07-27)
--------------------------

**Deobfuscation of the code base + pep8 and fixes**

-  Fixed ``tf.session().__enter__()`` being used, rather than
   ``sess = tf.session()`` and passing the session to the objects
-  Fixed uneven scoping of TensorFlow Sessions throughout the code
-  Fixed rolling vecwrapper to handle observations that are not only
   grayscale images
-  Fixed deepq saving the environment when trying to save itself
-  Fixed
   ``ValueError: Cannot take the length of Shape with unknown rank.`` in
   ``acktr``, when running ``run_atari.py`` script.
-  Fixed calling baselines sequentially no longer creates graph
   conflicts
-  Fixed mean on empty array warning with deepq
-  Fixed kfac eigen decomposition not cast to float64, when the
   parameter use_float64 is set to True
-  Fixed Dataset data loader, not correctly resetting id position if
   shuffling is disabled
-  Fixed ``EOFError`` when reading from connection in the ``worker`` in
   ``subproc_vec_env.py``
-  Fixed ``behavior_clone`` weight loading and saving for GAIL
-  Avoid taking root square of negative number in ``trpo_mpi.py``
-  Removed some duplicated code (a2cpolicy, trpo_mpi)
-  Removed unused, undocumented and crashing function ``reset_task`` in
   ``subproc_vec_env.py``
-  Reformated code to PEP8 style
-  Documented all the codebase
-  Added atari tests
-  Added logger tests

Missing: tests for acktr continuous (+ HER, gail but they rely on
mujoco...)
