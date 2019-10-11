RL Algorithms
=============

This table displays the rl algorithms that are implemented in the stable baselines project,
along with some useful characteristics: support for recurrent policies, discrete/continuous actions, multiprocessing.

.. Table too large
.. ===== ======================== ========= ======= ============ ================= =============== ================
.. Name  Refactored \ :sup:`(1)`\ Recurrent ``Box`` ``Discrete`` ``MultiDiscrete`` ``MultiBinary`` Multi Processing
.. ===== ======================== ========= ======= ============ ================= =============== ================
.. A2C   ✔️
.. ===== ======================== ========= ======= ============ ================= =============== ================


============ ======================== ========= =========== ============ ================
Name         Refactored [#f1]_        Recurrent ``Box``     ``Discrete`` Multi Processing
============ ======================== ========= =========== ============ ================
A2C          ✔️                        ✔️        ✔️           ✔️           ✔️
ACER         ✔️                        ✔️        ❌ [#f4]_    ✔️           ✔️
ACKTR        ✔️                        ✔️        ✔️            ✔️           ✔️
DDPG         ✔️                        ❌        ✔️           ❌           ✔️ [#f3]_
DQN          ✔️                        ❌        ❌           ✔️           ❌
HER          ✔️                        ❌        ✔️           ✔️           ❌
GAIL [#f2]_  ✔️                        ✔️        ✔️           ✔️           ✔️ [#f3]_
PPO1         ✔️                        ❌        ✔️           ✔️           ✔️ [#f3]_
PPO2         ✔️                        ✔️        ✔️           ✔️           ✔️
SAC          ✔️                        ❌        ✔️          ❌            ❌
TD3          ✔️                        ❌        ✔️          ❌            ❌
TRPO         ✔️                        ❌        ✔️           ✔            ✔️ [#f3]_
============ ======================== ========= =========== ============ ================

.. [#f1] Whether or not the algorithm has be refactored to fit the ``BaseRLModel`` class.
.. [#f2] Only implemented for TRPO.
.. [#f3] Multi Processing with `MPI`_.
.. [#f4] TODO, in project scope.

.. note::
    Non-array spaces such as ``Dict`` or ``Tuple`` are not currently supported by any algorithm,
    except HER for dict when working with ``gym.GoalEnv``

Actions ``gym.spaces``:

-  ``Box``: A N-dimensional box that containes every point in the action
   space.
-  ``Discrete``: A list of possible actions, where each timestep only
   one of the actions can be used.
-  ``MultiDiscrete``: A list of possible actions, where each timestep only one action of each discrete set can be used.
- ``MultiBinary``: A list of possible actions, where each timestep any of the actions can be used in any combination.

.. _MPI: https://mpi4py.readthedocs.io/en/stable/

.. note::

  Some logging values (like `ep_rewmean`, `eplenmean`) are only available when using a Monitor wrapper
  See `Issue #339 <https://github.com/hill-a/stable-baselines/issues/339>`_ for more info.


Reproducibility
---------------

Completely reproducible results are not guaranteed across Tensorflow releases or different platforms.
Furthermore, results need not be reproducible between CPU and GPU executions, even when using identical seeds.

In order to make make computations deterministic on CPU, on your specific problem on one specific platform,
you need to pass a `seed` argument at the creation of a model and set `n_cpu_tf_sess=1` (number of cpu for Tensorflow session).
If you pass an environment to the model using `set_env()`, then you also need to seed the environment first.

.. note::

  Because of the current limits of Tensorflow 1.x, we cannot ensure reproducible results on the GPU yet. We hope to solve that issue with Tensorflow 2.x support (cf `Issue #366 <https://github.com/hill-a/stable-baselines/issues/366>`_).


.. note::

  TD3 sometimes fail to have reproducible results for obscure reasons, even when following the previous steps (cf `PR #492 <https://github.com/hill-a/stable-baselines/pull/492>`_). If you find the reason then please open an issue ;)


Credit: part of the *Reproducibility* section comes from `PyTorch Documentation <https://pytorch.org/docs/stable/notes/randomness.html>`_
