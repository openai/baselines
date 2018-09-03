.. _install:

Installation
============

Prerequisites
-------------

Baselines requires python3 (>=3.5) with the development headers. You'll
also need system packages CMake, OpenMPI and zlib. Those can be
installed as follows

Ubuntu
~~~~~~

.. code-block:: bash

  sudo apt-get update && sudo apt-get install cmake libopenmpi-dev python3-dev zlib1g-dev

Mac OS X
~~~~~~~~

Installation of system packages on Mac requires `Homebrew`_. With
Homebrew installed, run the follwing:

.. code-block:: bash

   brew install cmake openmpi

.. _Homebrew: https://brew.sh


Stable Release
--------------

.. code-block:: bash

    pip install stable-baselines


Bleeding-edge version
---------------------

.. code-block:: bash

    git clone https://github.com/hill-a/stable-baselines && cd stable-baselines
    pip install -e .
