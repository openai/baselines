.. _changelog:

==========
Changelog
==========

For download links, please look at `Github release page <https://github.com/hill-a/stable-baselines/releases>`_.

Master version 1.0.8.a0 (WIP)
=============================

Nothing new for now...

Tensorboard support in progress (see ``tensorboard`` branch)


Release 1.0.7 (2018-08-29)
===========================

**Bug fixes and documentation**

- added html documentation using sphinx + integration with read the docs
- cleaned up README + typos
- fixed normalization for DQN with images
- fixed DQN identity test


Release 1.0.1 (2018-08-20)
==========================

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
==========================

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
