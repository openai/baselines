.. _gail:

.. automodule:: stable_baselines.gail


GAIL
====

`Generative Adversarial Imitation Learning (GAIL) <https://arxiv.org/abs/1606.03476>`_


Notes
-----

- Original paper: https://arxiv.org/abs/1606.03476

If you want to train an imitation learning agent
------------------------------------------------

.. _step-1:-download-expert-data:

Step 1: Download expert data
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Download the expert data into ``./data``, `download link`_

.. _step-2:-run-gail:

Step 2: Run GAIL
~~~~~~~~~~~~~~~~

Run with single thread:

.. code:: bash

   python -m stable_baselines.gail.run_mujoco

Run with multiple threads:

.. code:: bash

   mpirun -np 16 python -m stable_baselines.gail.run_mujoco

See help (``-h``) for more options.

.. _in-case-you-want-to-run-behavior-cloning-(bc):

**In case you want to run Behavior Cloning (BC)**

.. code:: bash

   python -m stable_baselines.gail.behavior_clone

See help (``-h``) for more options.


OpenAI Maintainers:

-  Yuan-Hong Liao, andrewliao11_at_gmail_dot_com
-  Ryan Julian, ryanjulian_at_gmail_dot_com

**Others**

Thanks to the open source:

-  @openai/imitation
-  @carpedm20/deep-rl-tensorflow

.. _download link: https://drive.google.com/drive/folders/1h3H4AY_ZBx08hz-Ct0Nxxus-V1melu1U?usp=sharing



Can I use?
----------

-  Reccurent policies: ✔️
-  Multi processing: ✔️ (using MPI)
-  Gym spaces:


============= ====== ===========
Space         Action Observation
============= ====== ===========
Discrete      ❌      ✔️
Box           ✔️       ✔️
MultiDiscrete ❌      ✔️
MultiBinary   ❌      ✔️
============= ====== ===========


Parameters
----------

.. autoclass:: GAIL
  :members:
  :inherited-members:
