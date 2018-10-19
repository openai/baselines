.. _vec_env:

.. automodule:: stable_baselines.common.vec_env

Vectorized Environments
=======================

Vectorized Environments are a way to multiprocess training. Instead of training a RL agent
on 1 environment, it allows to train it on `n` environments using `n` processes.
Because of that, `actions` passed to the environment are now a vector (of dimension `n`). It is the same for `observations`,
`rewards` and end of episode signals (`dones`).


.. note::

	Vectorized environments are required when using wrappers for frame-stacking or normalization.

.. note::

	When using vectorized environments, the environments are automatically resetted at the end of each episode.

.. warning::

	It seems that Windows users are experiencing issues with SubprocVecEnv.
	We recommend to use the docker image in that case. (See `Issue #42 <https://github.com/hill-a/stable-baselines/issues/40>`_)


DummyVecEnv
-----------

.. autoclass:: DummyVecEnv
  :members:

SubprocVecEnv
-------------

.. autoclass:: SubprocVecEnv
  :members:

Wrappers
--------

VecFrameStack
~~~~~~~~~~~~~

.. autoclass:: VecFrameStack
  :members:


VecNormalize
~~~~~~~~~~~~

.. autoclass:: VecNormalize
  :members:
