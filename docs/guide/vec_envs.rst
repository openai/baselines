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

		When using ``SubprocVecEnv``, Windows users must wrap the code
		in an ``if __name__=="__main__":``.
		See `stackoverflow question <https://stackoverflow.com/questions/24374288/where-to-put-freeze-support-in-a-python-script>`_
		for more information about multiprocessing on Windows using python.


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


VecVideoRecorder
~~~~~~~~~~~~~~~~

.. autoclass:: VecVideoRecorder
  :members:
