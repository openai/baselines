import numpy as np
import tensorflow as tf

import stable_baselines.common.tf_util as tf_util
from stable_baselines.common.distributions import DiagGaussianProbabilityDistributionType,\
    CategoricalProbabilityDistributionType, \
    MultiCategoricalProbabilityDistributionType, BernoulliProbabilityDistributionType


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

    nvec = np.array([1, 2, 3])
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
    tmp = pdparam + np.random.randn(pdparam.size) * 0.1
    mval2 = np.repeat(tmp[None, :], number_samples, axis=0)
    calckl = tf_util.function([mval_ph, mval2_ph], proba_distribution.kl(pd2))
    klval = calckl(mval, mval2).mean()
    logliks = calcloglik(xval, mval2)
    klval_ll = - entval - logliks.mean()
    klval_ll_stderr = logliks.std() / np.sqrt(number_samples)
    assert np.abs(klval - klval_ll) < 3 * klval_ll_stderr  # within 3 sigmas
    print('ok on', probtype, pdparam)
