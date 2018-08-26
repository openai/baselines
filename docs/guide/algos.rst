RL Algorithms
=============


.. Table too large
.. ===== ======================== ========= ======= ============ ================= =============== ================
.. Name  Refactored \ :sup:`(1)`\ Recurrent ``Box`` ``Discrete`` ``MultiDiscrete`` ``MultiBinary`` Multi Processing
.. ===== ======================== ========= ======= ============ ================= =============== ================
.. A2C   ✔️
.. ===== ======================== ========= ======= ============ ================= =============== ================

=========== ======================== ========= =========== ============ ================
Name        Refactored [#f1]_        Recurrent ``Box``     ``Discrete`` Multi Processing
=========== ======================== ========= =========== ============ ================
A2C         ✔️                        ✔️         ✔️           ✔️            ✔️
ACER        ✔️                        ✔️         ❌ [#f5]_   ✔️            ✔️
ACKTR       ✔️                        ✔️         ❌ [#f5]_   ✔️            ✔️
DDPG        ✔️                        ✔️         ✔️           ❌           ❌
DQN         ✔️                        ❌        ❌           ✔️            ❌
GAIL [#f2]_ ✔️                        ✔️         ✔️           ✔️            ✔️ [#f4]_
HER [#f3]_  ❌ [#f5]_                 ❌        ✔️           ❌           ❌
PPO1        ✔️                        ✔️         ✔️           ✔️            ✔️ [#f4]_
PPO2        ✔️                        ✔️         ✔️           ✔️            ✔️
TRPO        ✔️                        ✔️         ✔️           ✔️            ✔️ [#f4]_
=========== ======================== ========= =========== ============ ================

.. [#f1] Whether or not the algorithm has be refactored to fit the ``BaseRLModel`` class.
.. [#f2] Only implemented for TRPO.
.. [#f3] Only implemented for DDPG.
.. [#f4] Multi Processing with `MPI`_.
.. [#f5] TODO, in project scope.


Actions ``gym.spaces``:

-  ``Box``: A N-dimensional box that containes every point in the action
   space.
-  ``Discrete``: A list of possible actions, where each timestep only
   one of the actions can be used.
-  ``MultiDiscrete``: A list of possible actions, where each timestep only one action of each discrete set can be used.
- ``MultiBinary``: A list of possible actions, where each timestep any of the actions can be used in any combination.

.. _MPI: https://mpi4py.readthedocs.io/en/stable/
