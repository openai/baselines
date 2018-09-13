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


Using Docker Images
-------------------

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
