.. _vec_env:

.. automodule:: stable_baselines.common.vec_env

Vectorized Environments
=======================

Vectorized Environments are a method for multiprocess training. Instead of training an RL agent
on 1 environment, it allows us to train it on `n` environments using `n` processes.
Because of this, `actions` passed to the environment are now a vector (of dimension `n`).
It is the same for `observations`, `rewards` and end of episode signals (`dones`).
In the case of non-array observation spaces such as `Dict` or `Tuple`, where different sub-spaces
may have different shapes, the sub-observations are vectors (of dimension `n`).

============= ======= ============ ======== ========= ================
Name          ``Box`` ``Discrete`` ``Dict`` ``Tuple`` Multi Processing
============= ======= ============ ======== ========= ================
DummyVecEnv   ✔️       ✔️           ✔️        ✔️         ❌️
SubprocVecEnv ✔️       ✔️           ✔️        ✔️         ✔️
============= ======= ============ ======== ========= ================

.. note::

	Vectorized environments are required when using wrappers for frame-stacking or normalization.

.. note::

	When using vectorized environments, the environments are automatically reset at the end of each episode.

.. warning::

				When using ``SubprocVecEnv``, users must wrap the code in an ``if __name__ == "__main__":`` if using the ``forkserver`` or ``spawn`` start method (default on Windows).
				On Linux, the default start method is ``fork`` which is not thread safe and can create deadlocks.

				For more information, see Python's `multiprocessing guidelines <https://docs.python.org/3/library/multiprocessing.html#the-spawn-and-forkserver-start-methods>`_.


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
