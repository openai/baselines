.. _distributions:

Probability Distributions
=========================

Probability distributions used for the different action spaces:

- ``CategoricalProbabilityDistribution`` -> Discrete
- ``DiagGaussianProbabilityDistribution`` -> Box (continuous actions)
- ``MultiCategoricalProbabilityDistribution`` -> MultiDiscrete
- ``BernoulliProbabilityDistribution`` -> MultiBinary

The policy networks output parameters for the distributions (named `flat` in the methods).
Actions are then sampled from those distributions.

For instance, in the case of discrete actions. The policy network outputs probability
of taking each action. The ``CategoricalProbabilityDistribution`` allows to sample from it,
computes the entropy, the negative log probability (``neglogp``) and backpropagate the gradient.

In the case of continuous actions, a Gaussian distribution is used. The policy network outputs
mean and (log) std of the distribution (assumed to be a ``DiagGaussianProbabilityDistribution``).

.. automodule:: stable_baselines.common.distributions
  :members:
