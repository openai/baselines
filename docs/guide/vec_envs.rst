.. _vec_env:

.. automodule:: stable_baselines.common.vec_env

Vectorized Environments
=======================

Vectorized Environments are a method for stacking multiple independent environments into a single environment.
Instead of training an RL agent on 1 environment per step, it allows us to train it on `n` environments per step.
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
	Thus, the observation returned for the i-th environment when ``done[i]`` is true will in fact be the first observation of the next episode, not the last observation of the episode that has just terminated.
	You can access the "real" final observation of the terminated episode—that is, the one that accompanied the ``done`` event provided by the underlying environment—using the ``terminal_observation`` keys in the info dicts returned by the vecenv.

.. warning::

				When using ``SubprocVecEnv``, users must wrap the code in an ``if __name__ == "__main__":`` if using the ``forkserver`` or ``spawn`` start method (default on Windows).
				On Linux, the default start method is ``fork`` which is not thread safe and can create deadlocks.

				For more information, see Python's `multiprocessing guidelines <https://docs.python.org/3/library/multiprocessing.html#the-spawn-and-forkserver-start-methods>`_.

VecEnv
------

.. autoclass:: VecEnv
  :members:

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


VecCheckNan
~~~~~~~~~~~~~~~~

.. autoclass:: VecCheckNan
  :members:
