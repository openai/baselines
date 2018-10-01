.. _policies:

.. automodule:: stable_baselines.common.policies

Policy Networks
===============

.. warning::
  For all algorithms (except DDPG), continuous actions are only clipped during training
  (to avoid out of bound error). However, you have to manually clip the action when using
  the `predict()` method.


Base Classes
------------

.. autoclass:: ActorCriticPolicy
  :members:

.. autoclass:: FeedForwardPolicy
  :members:

.. autoclass:: LstmPolicy
  :members:

MLP Policies
------------

.. autoclass:: MlpPolicy
  :members:

.. autoclass:: MlpLstmPolicy
  :members:

.. autoclass:: MlpLnLstmPolicy
  :members:


CNN Policies
------------

.. autoclass:: CnnPolicy
  :members:

.. autoclass:: CnnLstmPolicy
  :members:

.. autoclass:: CnnLnLstmPolicy
  :members:
