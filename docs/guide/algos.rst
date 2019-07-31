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
ACKTR        ✔️                        ✔️        ❌ [#f4]_    ✔️           ✔️
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
