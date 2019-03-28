.. Stable Baselines documentation master file, created by
   sphinx-quickstart on Sat Aug 25 10:33:54 2018.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to Stable Baselines docs! - RL Baselines Made Easy
===========================================================

`Stable Baselines <https://github.com/hill-a/stable-baselines>`_ is a set of improved implementations
of Reinforcement Learning (RL) algorithms based on OpenAI `Baselines <https://github.com/openai/baselines>`_.

Github repository: https://github.com/hill-a/stable-baselines

RL Baselines Zoo (collection of pre-trained agents): https://github.com/araffin/rl-baselines-zoo

RL Baselines zoo also offers a simple interface to train and evaluate agents.

You can read a detailed presentation of Stable Baselines in the
Medium article: `link <https://medium.com/@araffin/stable-baselines-a-fork-of-openai-baselines-reinforcement-learning-made-easy-df87c4b2fc82>`_


.. Example of internal link: :ref:`ppo2`

Main differences with OpenAI Baselines
--------------------------------------

This toolset is a fork of OpenAI Baselines, with a major structural refactoring, and code cleanups:

- Unified structure for all algorithms
- PEP8 compliant (unified code style)
- Documented functions and classes
- More tests & more code coverage


.. toctree::
   :maxdepth: 2
   :caption: User Guide

   guide/install
   guide/quickstart
   guide/rl
   guide/algos
   guide/examples
   guide/vec_envs
   guide/custom_env
   guide/custom_policy
   guide/tensorboard
   guide/rl_zoo
   guide/pretrain


.. toctree::
  :maxdepth: 1
  :caption: RL Algorithms

  modules/base
  modules/policies
  modules/a2c
  modules/acer
  modules/acktr
  modules/ddpg
  modules/dqn
  modules/gail
  modules/her
  modules/ppo1
  modules/ppo2
  modules/sac
  modules/trpo

.. toctree::
  :maxdepth: 1
  :caption: Common

  common/distributions
  common/tf_utils
  common/cmd_utils
  common/schedules

.. toctree::
  :maxdepth: 1
  :caption: Misc

  misc/changelog
  misc/projects
  misc/results_plotter


Citing Stable Baselines
-----------------------
To cite this project in publications:

.. code-block:: bibtex

    @misc{stable-baselines,
      author = {Hill, Ashley and Raffin, Antonin and Ernestus, Maximilian and Gleave, Adam and Traore, Rene and Dhariwal, Prafulla and Hesse, Christopher and Klimov, Oleg and Nichol, Alex and Plappert, Matthias and Radford, Alec and Schulman, John and Sidor, Szymon and Wu, Yuhuai},
      title = {Stable Baselines},
      year = {2018},
      publisher = {GitHub},
      journal = {GitHub repository},
      howpublished = {\url{https://github.com/hill-a/stable-baselines}},
    }

Contributing
------------

To any interested in making the rl baselines better, there is still some improvements
that needs to be done.
A full TODO list is available in the `roadmap <https://github.com/hill-a/stable-baselines/projects/1>`_.

If you want to contribute, please read `CONTRIBUTING.md <https://github.com/hill-a/stable-baselines/blob/master/CONTRIBUTING.md>`_ first.

Indices and tables
-------------------

* :ref:`genindex`
* :ref:`search`
* :ref:`modindex`
