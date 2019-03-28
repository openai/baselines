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


Windows 10
~~~~~~~~~~

We recommend using `Anaconda <https://conda.io/docs/user-guide/install/windows.html>`_ for windows users.

0. Create a new environment in the Anaconda Navigator (at least python 3.5) and install ``zlib`` in this environment.

1. Install `MPI for Windows <https://www.microsoft.com/en-us/download/details.aspx?id=57467>`_ (you need to download and install ``msmpisetup.exe``)

2. Clone Stable-Baselines Github repo and replace the line ``gym[atari,classic_control]>=0.10.9`` in ``setup.py`` by this one: ``gym[classic_control]>=0.10.9``

3. Install Stable-Baselines from source, inside the folder, run ``pip install -e .``

4. [Optional] If you want to use atari environments, you need to install this package: https://github.com/j8lp/atari-py
(using again ``pip install -e .``)


Stable Release
--------------

.. code-block:: bash

    pip install stable-baselines


Bleeding-edge version
---------------------

With support for running tests and building the documentation.

.. code-block:: bash

    git clone https://github.com/hill-a/stable-baselines && cd stable-baselines
    pip install -e .[docs,tests]


Using Docker Images
-------------------

If you are looking for docker images with stable-baselines already installed in it,
we recommend using images from `RL Baselines Zoo <https://github.com/araffin/rl-baselines-zoo>`_.

Otherwise, the following images contained all the dependencies for stable-baselines but not the stable-baselines package itself.
They are made for development.

Use Built Images
~~~~~~~~~~~~~~~~

GPU image (requires `nvidia-docker`_):

.. code-block:: bash

   docker pull araffin/stable-baselines

CPU only:

.. code-block:: bash

   docker pull araffin/stable-baselines-cpu

Build the Docker Images
~~~~~~~~~~~~~~~~~~~~~~~~

Build GPU image (with nvidia-docker):

.. code-block:: bash

   docker build . -f docker/Dockerfile.gpu -t stable-baselines

Build CPU image:

.. code-block:: bash

   docker build . -f docker/Dockerfile.cpu -t stable-baselines-cpu

Note: if you are using a proxy, you need to pass extra params during
build and do some `tweaks`_:

.. code-block:: bash

   --network=host --build-arg HTTP_PROXY=http://your.proxy.fr:8080/ --build-arg http_proxy=http://your.proxy.fr:8080/ --build-arg HTTPS_PROXY=https://your.proxy.fr:8080/ --build-arg https_proxy=https://your.proxy.fr:8080/

Run the images (CPU/GPU)
~~~~~~~~~~~~~~~~~~~~~~~~

Run the nvidia-docker GPU image

.. code-block:: bash

   docker run -it --runtime=nvidia --rm --network host --ipc=host --name test --mount src="$(pwd)",target=/root/code/stable-baselines,type=bind araffin/stable-baselines bash -c 'cd /root/code/stable-baselines/ && pytest tests/'

Or, with the shell file:

.. code-block:: bash

   ./run_docker_gpu.sh pytest tests/

Run the docker CPU image

.. code-block:: bash

   docker run -it --rm --network host --ipc=host --name test --mount src="$(pwd)",target=/root/code/stable-baselines,type=bind araffin/stable-baselines-cpu bash -c 'cd /root/code/stable-baselines/ && pytest tests/'

Or, with the shell file:

.. code-block:: bash

   ./run_docker_cpu.sh pytest tests/

Explanation of the docker command:

-  ``docker run -it`` create an instance of an image (=container), and
   run it interactively (so ctrl+c will work)
-  ``--rm`` option means to remove the container once it exits/stops
   (otherwise, you will have to use ``docker rm``)
-  ``--network host`` don't use network isolation, this allow to use
   tensorboard/visdom on host machine
-  ``--ipc=host`` Use the host system’s IPC namespace. IPC (POSIX/SysV IPC) namespace provides
   separation of named shared memory segments, semaphores and message
   queues.
-  ``--name test`` give explicitely the name ``test`` to the container,
   otherwise it will be assigned a random name
-  ``--mount src=...`` give access of the local directory (``pwd``
   command) to the container (it will be map to ``/root/code/stable-baselines``), so
   all the logs created in the container in this folder will be kept
-  ``bash -c '...'`` Run command inside the docker image, here run the tests
   (``pytest tests/``)

.. _nvidia-docker: https://github.com/NVIDIA/nvidia-docker
.. _tweaks: https://stackoverflow.com/questions/23111631/cannot-download-docker-images-behind-a-proxy
