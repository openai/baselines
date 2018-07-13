import tensorflow as tf
from tensorflow.python.ops import math_ops
import numpy as np
from gym import spaces

import baselines.common.tf_util as tf_util
from baselines.a2c.utils import linear


class ProbabilityDistribution(object):
    """
    A particular probability distribution
    """
    def flatparam(self):
        """
        Return the direct probabilities

        :return: ([float]) the probabilites
        """
        raise NotImplementedError

    def mode(self):
        """
        Returns the index of the highest probability

        :return: (int) the max index of the probabilites
        """
        raise NotImplementedError

    def neglogp(self, x):
        """
        returns the of the negative log likelihood

        :param x: (str) the labels of each index
        :return: ([float]) The negative log likelihood of the distribution
        """
        # Usually it's easier to define the negative logprob
        raise NotImplementedError

    def kl(self, other):
        """
        Calculates the Kullback-Leiber divergence from the given probabilty distribution

        :param other: ([float]) the distibution to compare with
        :return: (float) the KL divergence of the two distributions
        """
        raise NotImplementedError

    def entropy(self):
        """
        Returns shannon's entropy of the probability

        :return: (float) the entropy
        """
        raise NotImplementedError

    def sample(self):
        """
        Sample an index from the probabilty distribution

        :return: (int) the sampled index
        """
        raise NotImplementedError

    def logp(self, x):
        """
        returns the of the log likelihood

        :param x: (str) the labels of each index
        :return: ([float]) The log likelihood of the distribution
        """
        return - self.neglogp(x)


class ProbabilityDistributionType(object):
    """
    Parametrized family of probability distributions
    """
    def probability_distribution_class(self):
        """
        returns the ProbabilityDistribution class of this type

        :return: (Type ProbabilityDistribution) the probability distribution class associated
        """
        raise NotImplementedError

    def proba_distribution_from_flat(self, flat):
        """
        returns the probability distribution from flat probabilities

        :param flat: ([float]) the flat probabilities
        :return: (ProbabilityDistribution) the instance of the ProbabilityDistribution associated
        """
        return self.probability_distribution_class()(flat)

    def proba_distribution_from_latent(self, latent_vector):
        """
        returns the probability distribution from latent values

        :param latent_vector: ([float]) the latent values
        :return: (ProbabilityDistribution) the instance of the ProbabilityDistribution associated
        """
        raise NotImplementedError

    def param_shape(self):
        """
        returns the shape of the input parameters

        :return: ([int]) the shape
        """
        raise NotImplementedError

    def sample_shape(self):
        """
        returns the shape of the sampling

        :return: ([int]) the shape
        """
        raise NotImplementedError

    def sample_dtype(self):
        """
        returns the type of the sampling

        :return: (type) the type
        """
        raise NotImplementedError

    def param_placeholder(self, prepend_shape, name=None):
        """
        returns the TensorFlow placeholder for the input parameters

        :param prepend_shape: ([int]) the prepend shape
        :param name: (str) the placeholder name
        :return: (TensorFlow Tensor) the placeholder
        """
        return tf.placeholder(dtype=tf.float32, shape=prepend_shape+self.param_shape(), name=name)

    def sample_placeholder(self, prepend_shape, name=None):
        """
        returns the TensorFlow placeholder for the sampling

        :param prepend_shape: ([int]) the prepend shape
        :param name: (str) the placeholder name
        :return: (TensorFlow Tensor) the placeholder
        """
        return tf.placeholder(dtype=self.sample_dtype(), shape=prepend_shape+self.sample_shape(), name=name)


class CategoricalProbabilityDistributionType(ProbabilityDistributionType):
    def __init__(self, ncat):
        """
        The probability distribution type for categorical input

        :param ncat: (int) the number of categories
        """
        self.ncat = ncat

    def probability_distribution_class(self):
        return CategoricalProbabilityDistribution

    def proba_distribution_from_latent(self, latent_vector, init_scale=1.0, init_bias=0.0):
        """
        returns the probability distribution from latent values

        :param latent_vector: ([float]) the latent values
        :param init_scale: (float) the inital scale of the distribution
        :param init_bias: (float) the inital bias of the distribution
        :return: (ProbabilityDistribution) the instance of the ProbabilityDistribution associated
        """
        pdparam = linear(latent_vector, 'pi', self.ncat, init_scale=init_scale, init_bias=init_bias)
        return self.proba_distribution_from_flat(pdparam), pdparam

    def param_shape(self):
        return [self.ncat]

    def sample_shape(self):
        return []

    def sample_dtype(self):
        return tf.int32


class MultiCategoricalProbabilityDistributionType(ProbabilityDistributionType):
    def __init__(self, nvec):
        """
        The probability distribution type for multiple categorical input

        :param nvec: (int) the number of vectors
        """
        self.ncats = nvec

    def probability_distribution_class(self):
        return MultiCategoricalProbabilityDistribution

    def proba_distribution_from_flat(self, flat):
        return MultiCategoricalProbabilityDistribution(self.ncats, flat)

    def proba_distribution_from_latent(self, latent_vector):
        raise NotImplementedError

    def param_shape(self):
        return [sum(self.ncats)]

    def sample_shape(self):
        return [len(self.ncats)]

    def sample_dtype(self):
        return tf.int32


class DiagGaussianProbabilityDistributionType(ProbabilityDistributionType):
    def __init__(self, size):
        """
        The probability distribution type for multivariate gaussian input

        :param size: (int) the number of dimentions of the multivariate gaussian
        """
        self.size = size

    def probability_distribution_class(self):
        return DiagGaussianProbabilityDistribution

    def proba_distribution_from_latent(self, latent_vector, init_scale=1.0, init_bias=0.0):
        """
        returns the probability distribution from latent values

        :param latent_vector: ([float]) the latent values
        :param init_scale: (float) the inital scale of the distribution
        :param init_bias: (float) the inital bias of the distribution
        :return: (ProbabilityDistribution) the instance of the ProbabilityDistribution associated
        """
        mean = linear(latent_vector, 'pi', self.size, init_scale=init_scale, init_bias=init_bias)
        logstd = tf.get_variable(name='logstd', shape=[1, self.size], initializer=tf.zeros_initializer())
        pdparam = tf.concat([mean, mean * 0.0 + logstd], axis=1)
        return self.proba_distribution_from_flat(pdparam), mean

    def param_shape(self):
        return [2*self.size]

    def sample_shape(self):
        return [self.size]

    def sample_dtype(self):
        return tf.float32


class BernoulliProbabilityDistributionType(ProbabilityDistributionType):
    def __init__(self, size):
        """
        The probability distribution type for bernoulli input

        :param size: (int) the number of dimentions of the bernoulli distribution
        """
        self.size = size

    def probability_distribution_class(self):
        return BernoulliProbabilityDistribution

    def proba_distribution_from_latent(self, latent_vector):
        raise NotImplementedError

    def param_shape(self):
        return [self.size]

    def sample_shape(self):
        return [self.size]

    def sample_dtype(self):
        return tf.int32


class CategoricalProbabilityDistribution(ProbabilityDistribution):
    def __init__(self, logits):
        """
        Probability distributions from categorical input

        :param logits: ([float]) the categorical logits input
        """
        self.logits = logits

    def flatparam(self):
        return self.logits

    def mode(self):
        return tf.argmax(self.logits, axis=-1)

    def neglogp(self, x):
        # return tf.nn. (logits=self.logits, labels=x)
        # Note: we can't use sparse_softmax_cross_entropy_with_logits because
        #       the implementation does not allow second-order derivatives...
        one_hot_actions = tf.one_hot(x, self.logits.get_shape().as_list()[-1])
        return tf.nn.softmax_cross_entropy_with_logits(
            logits=self.logits,
            labels=one_hot_actions)

    def kl(self, other):
        a0 = self.logits - tf.reduce_max(self.logits, axis=-1, keep_dims=True)
        a1 = other.logits - tf.reduce_max(other.logits, axis=-1, keep_dims=True)
        ea0 = tf.exp(a0)
        ea1 = tf.exp(a1)
        z0 = tf.reduce_sum(ea0, axis=-1, keep_dims=True)
        z1 = tf.reduce_sum(ea1, axis=-1, keep_dims=True)
        p0 = ea0 / z0
        return tf.reduce_sum(p0 * (a0 - tf.log(z0) - a1 + tf.log(z1)), axis=-1)

    def entropy(self):
        a0 = self.logits - tf.reduce_max(self.logits, axis=-1, keep_dims=True)
        ea0 = tf.exp(a0)
        z0 = tf.reduce_sum(ea0, axis=-1, keep_dims=True)
        p0 = ea0 / z0
        return tf.reduce_sum(p0 * (tf.log(z0) - a0), axis=-1)

    def sample(self):
        u = tf.random_uniform(tf.shape(self.logits))
        return tf.argmax(self.logits - tf.log(-tf.log(u)), axis=-1)

    @classmethod
    def fromflat(cls, flat):
        """
        Create an instance of this from new logits values

        :param flat: ([float]) the categorical logits input
        :return: (ProbabilityDistribution) the instance from the given categorical input
        """
        return cls(flat)


class MultiCategoricalProbabilityDistribution(ProbabilityDistribution):
    def __init__(self, nvec, flat):
        """
        Probability distributions from multicategorical input

        :param nvec: (int) the number of categorical inputs
        :param flat: ([float]) the categorical logits input
        """
        self.flat = flat
        self.categoricals = list(map(CategoricalProbabilityDistribution, tf.split(flat, nvec, axis=-1)))

    def flatparam(self):
        return self.flat

    def mode(self):
        return tf.cast(tf.stack([p.mode() for p in self.categoricals], axis=-1), tf.int32)

    def neglogp(self, x):
        return tf.add_n([p.neglogp(px) for p, px in zip(self.categoricals, tf.unstack(x, axis=-1))])

    def kl(self, other):
        return tf.add_n([p.kl(q) for p, q in zip(self.categoricals, other.categoricals)])

    def entropy(self):
        return tf.add_n([p.entropy() for p in self.categoricals])

    def sample(self):
        return tf.cast(tf.stack([p.sample() for p in self.categoricals], axis=-1), tf.int32)

    @classmethod
    def fromflat(cls, flat):
        """
        Create an instance of this from new logits values

        :param flat: ([float]) the multi categorical logits input
        :return: (ProbabilityDistribution) the instance from the given multi categorical input
        """
        raise NotImplementedError


class DiagGaussianProbabilityDistribution(ProbabilityDistribution):
    def __init__(self, flat):
        """
        Probability distributions from multivariate gaussian input

        :param flat: ([float]) the multivariate gaussian input data
        """
        self.flat = flat
        mean, logstd = tf.split(axis=len(flat.shape)-1, num_or_size_splits=2, value=flat)
        self.mean = mean
        self.logstd = logstd
        self.std = tf.exp(logstd)

    def flatparam(self):
        return self.flat

    def mode(self):
        return self.mean

    def neglogp(self, x):
        return 0.5 * tf.reduce_sum(tf.square((x - self.mean) / self.std), axis=-1) \
               + 0.5 * np.log(2.0 * np.pi) * tf.to_float(tf.shape(x)[-1]) \
               + tf.reduce_sum(self.logstd, axis=-1)

    def kl(self, other):
        assert isinstance(other, DiagGaussianProbabilityDistribution)
        return tf.reduce_sum(other.logstd - self.logstd + (tf.square(self.std) + tf.square(self.mean - other.mean)) /
                             (2.0 * tf.square(other.std)) - 0.5, axis=-1)

    def entropy(self):
        return tf.reduce_sum(self.logstd + .5 * np.log(2.0 * np.pi * np.e), axis=-1)

    def sample(self):
        return self.mean + self.std * tf.random_normal(tf.shape(self.mean))

    @classmethod
    def fromflat(cls, flat):
        """
        Create an instance of this from new multivariate gaussian input

        :param flat: ([float]) the multivariate gaussian input data
        :return: (ProbabilityDistribution) the instance from the given multivariate gaussian input data
        """
        return cls(flat)


class BernoulliProbabilityDistribution(ProbabilityDistribution):
    def __init__(self, logits):
        """
        Probability distributions from bernoulli input

        :param logits: ([float]) the bernoulli input data
        """
        self.logits = logits
        self.ps = tf.sigmoid(logits)

    def flatparam(self):
        return self.logits

    def mode(self):
        return tf.round(self.ps)

    def neglogp(self, x):
        return tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.logits, labels=tf.to_float(x)),
                             axis=-1)

    def kl(self, other):
        return tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(logits=other.logits, labels=self.ps), axis=-1) - \
            tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.logits, labels=self.ps), axis=-1)

    def entropy(self):
        return tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.logits, labels=self.ps), axis=-1)

    def sample(self):
        u = tf.random_uniform(tf.shape(self.ps))
        return tf.to_float(math_ops.less(u, self.ps))

    @classmethod
    def fromflat(cls, flat):
        """
        Create an instance of this from new bernoulli input

        :param flat: ([float]) the bernoulli input data
        :return: (ProbabilityDistribution) the instance from the given bernoulli input data
        """
        return cls(flat)


def make_proba_dist_type(ac_space):
    """
    return an instance of ProbabilityDistributionType for the correct type of action space

    :param ac_space: (Gym Space) the input action space
    :return: (ProbabilityDistributionType) the approriate instance of a ProbabilityDistributionType
    """
    if isinstance(ac_space, spaces.Box):
        assert len(ac_space.shape) == 1
        return DiagGaussianProbabilityDistributionType(ac_space.shape[0])
    elif isinstance(ac_space, spaces.Discrete):
        return CategoricalProbabilityDistributionType(ac_space.n)
    elif isinstance(ac_space, spaces.MultiDiscrete):
        return MultiCategoricalProbabilityDistributionType(ac_space.nvec)
    elif isinstance(ac_space, spaces.MultiBinary):
        return BernoulliProbabilityDistributionType(ac_space.n)
    else:
        raise NotImplementedError


def shape_el(v, i):
    """
    get the shape of a TensorFlow Tensor element

    :param v: (TensorFlow Tensor) the input tensor
    :param i: (int) the element
    :return: ([int]) the shape
    """
    maybe = v.get_shape()[i]
    if maybe is not None:
        return maybe
    else:
        return tf.shape(v)[i]


@tf_util.in_session
def test_probtypes():
    """
    test probability distribution types
    """
    np.random.seed(0)

    pdparam_diag_gauss = np.array([-.2, .3, .4, -.5, .1, -.5, .1, 0.8])
    diag_gauss = DiagGaussianProbabilityDistributionType(pdparam_diag_gauss.size // 2)
    validate_probtype(diag_gauss, pdparam_diag_gauss)

    pdparam_categorical = np.array([-.2, .3, .5])
    categorical = CategoricalProbabilityDistributionType(pdparam_categorical.size)
    validate_probtype(categorical, pdparam_categorical)

    nvec = [1, 2, 3]
    pdparam_multicategorical = np.array([-.2, .3, .5, .1, 1, -.1])
    multicategorical = MultiCategoricalProbabilityDistributionType(nvec)
    validate_probtype(multicategorical, pdparam_multicategorical)

    pdparam_bernoulli = np.array([-.2, .3, .5])
    bernoulli = BernoulliProbabilityDistributionType(pdparam_bernoulli.size)
    validate_probtype(bernoulli, pdparam_bernoulli)


def validate_probtype(probtype, pdparam):
    """
    validate probability distribution types

    :param probtype: (ProbabilityDistributionType) the type to validate
    :param pdparam: ([float]) the flat probabilities to test
    """
    number_samples = 100000
    # Check to see if mean negative log likelihood == differential entropy
    mval = np.repeat(pdparam[None, :], number_samples, axis=0)
    mval_ph = probtype.param_placeholder([number_samples])
    xval_ph = probtype.sample_placeholder([number_samples])
    proba_distribution = probtype.proba_distribution_from_flat(mval_ph)
    calcloglik = tf_util.function([xval_ph, mval_ph], proba_distribution.logp(xval_ph))
    calcent = tf_util.function([mval_ph], proba_distribution.entropy())
    xval = tf.get_default_session().run(proba_distribution.sample(), feed_dict={mval_ph: mval})
    logliks = calcloglik(xval, mval)
    entval_ll = - logliks.mean()
    entval_ll_stderr = logliks.std() / np.sqrt(number_samples)
    entval = calcent(mval).mean()
    assert np.abs(entval - entval_ll) < 3 * entval_ll_stderr  # within 3 sigmas

    # Check to see if kldiv[p,q] = - ent[p] - E_p[log q]
    mval2_ph = probtype.param_placeholder([number_samples])
    pd2 = probtype.proba_distribution_from_flat(mval2_ph)
    q = pdparam + np.random.randn(pdparam.size) * 0.1
    mval2 = np.repeat(q[None, :], number_samples, axis=0)
    calckl = tf_util.function([mval_ph, mval2_ph], proba_distribution.kl(pd2))
    klval = calckl(mval, mval2).mean()
    logliks = calcloglik(xval, mval2)
    klval_ll = - entval - logliks.mean()
    klval_ll_stderr = logliks.std() / np.sqrt(number_samples)
    assert np.abs(klval - klval_ll) < 3 * klval_ll_stderr  # within 3 sigmas
    print('ok on', probtype, pdparam)
