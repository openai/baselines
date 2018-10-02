.. _policies:

.. automodule:: stable_baselines.common.policies

Policy Networks
===============

Stable-baselines provides a set of default policies, that can be used with most action spaces.
If you need more control on the policy architecture, You can also create a custom policy (see :ref:`custom_policy`).

.. note::

	CnnPolicies are for images only. MlpPolicies are made for other type of features (e.g. robot joints)

.. warning::
  For all algorithms (except DDPG), continuous actions are only clipped during training
  (to avoid out of bound error). However, you have to manually clip the action when using
  the `predict()` method.

.. rubric:: Available Policies

.. autosummary::
    :nosignatures:

    MlpPolicy
    MlpLstmPolicy
    MlpLnLstmPolicy
    CnnPolicy
    CnnLstmPolicy
    CnnLnLstmPolicy


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
