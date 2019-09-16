.. _export:


Exporting models
================

After training an agent, you may want to deploy/use it in an other language
or framework, like PyTorch or `tensorflowjs <https://github.com/tensorflow/tfjs>`_.
Stable Baselines does not include tools to export models to other frameworks, but
this document aims to cover parts that are required for exporting along with
more detailed stories from users of Stable Baselines.


Background
----------

In Stable Baselines, the controller is stored inside :ref:`policies <policies>` which convert
observations into actions. Each learning algorithm (e.g. DQN, A2C, SAC) contains
one or more policies, some of which are only used for training. An easy way to find
the policy is to check the code for the ``predict`` function of the agent:
This function should only call one policy with simple arguments.

Policies hold the necessary Tensorflow placeholders and tensors to do the
inference (i.e. predict actions), so it is enough to export these policies
to do inference in an another framework.

.. note::
  Learning algorithms also may contain other Tensorflow placeholders, that are used for training only and are
  not required for inference.


.. warning::
  When using CNN policies, the observation is normalized internally (dividing by 255 to have values in [0, 1])


Export to PyTorch
-----------------

A known working solution is to use :func:`get_parameters <stable_baselines.common.base_class.BaseRLModel.get_parameters>`
function to obtain model parameters, construct the network manually in PyTorch and assign parameters correctly.

.. warning::
  PyTorch and Tensorflow have internal differences with e.g. 2D convolutions (see discussion linked below).


See `discussion #372 <https://github.com/hill-a/stable-baselines/issues/372>`_ for details.


Export to tensorflowjs / tfjs
-----------------------------

Can be done via Tensorflow's `simple_save <https://www.tensorflow.org/api_docs/python/tf/saved_model/simple_save>`_ function
and `tensorflowjs_converter <https://www.tensorflow.org/js/tutorials/conversion/import_saved_model>`_.

See `discussion #474 <https://github.com/hill-a/stable-baselines/issues/474>`_ for details.


Export to Java
---------------

Can be done via Tensorflow's `simple_save <https://www.tensorflow.org/api_docs/python/tf/saved_model/simple_save>`_ function.

See `this discussion <https://github.com/hill-a/stable-baselines/issues/329>`_ for details.


Manual export
-------------

You can also manually export required parameters (weights) and construct the
network in your desired framework, as done with the PyTorch example above.

You can access parameters of the model via agents'
:func:`get_parameters <stable_baselines.common.base_class.BaseRLModel.get_parameters>`
function. If you use default policies, you can find the architecture of the networks in
source for :ref:`policies <policies>`. Otherwise, for DQN/SAC/DDPG or TD3 you need to check the `policies.py` file located
in their respective folders.
