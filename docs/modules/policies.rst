.. _policies:

.. automodule:: stable_baselines.common.policies

Policy Networks
===============

Stable-baselines provides a set of default policies, that can be used with most action spaces.
To customize the default policies, you can specify the ``policy_kwargs`` parameter to the model class you use.
Those kwargs are then passed to the policy on instantiation (see :ref:`custom_policy` for an example).
If you need more control on the policy architecture, you can also create a custom policy (see :ref:`custom_policy`).

.. note::

	CnnPolicies are for images only. MlpPolicies are made for other type of features (e.g. robot joints)

.. warning::
  For all algorithms (except DDPG and SAC), continuous actions are clipped during training and testing
  (to avoid out of bound error).


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
