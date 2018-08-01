import tensorflow as tf
import numpy as np
from baselines.common.tf_util import get_session

class RunningMeanStd(object):
    # https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm
    def __init__(self, epsilon=1e-4, shape=()):
        self.mean = np.zeros(shape, 'float64')
        self.var = np.ones(shape, 'float64')
        self.count = epsilon

    def update(self, x):
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean, batch_var, batch_count):
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / tot_count        
        m_a = self.var * (self.count)
        m_b = batch_var * (batch_count)
        M2 = m_a + m_b + np.square(delta) * self.count * batch_count / (self.count + batch_count)
        new_var = M2 / (self.count + batch_count)

        new_count = batch_count + self.count

        self.mean = new_mean
        self.var = new_var
        self.count = new_count    


class TfRunningMeanStd(object):
    # https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm
    '''
    TensorFlow variables-based implmentation of computing running mean and std
    Benefit of this implementation is that it can be saved / loaded together with the tensorflow model
    '''
    def __init__(self, epsilon=1e-4, shape=(), scope=''):
        sess = get_session()

        _batch_mean = tf.placeholder(shape=shape, dtype=tf.float64)
        _batch_var = tf.placeholder(shape=shape, dtype=tf.float64)
        _batch_count = tf.placeholder(shape=(), dtype=tf.float64)

        
        with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
            _mean  = tf.get_variable('mean',  initializer=np.zeros(shape, 'float64'),     dtype=tf.float64)
            _var   = tf.get_variable('std',   initializer=np.ones(shape, 'float64'),      dtype=tf.float64)    
            _count = tf.get_variable('count', initializer=np.ones((), 'float64')*epsilon, dtype=tf.float64)

        delta = _batch_mean - _mean
        tot_count = _count + _batch_count

        new_mean = _mean + delta * _batch_count / tot_count        
        m_a = _var * (_count)
        m_b = _batch_var * (_batch_count)
        M2 = m_a + m_b + np.square(delta) * _count * _batch_count / (_count + _batch_count)
        new_var = M2 / (_count + _batch_count)
        new_count = _batch_count + _count

        update_ops = [
            _var.assign(new_var),
            _mean.assign(new_mean),
            _count.assign(new_count)
        ]

        self._mean = _mean
        self._var = _var
        self._count = _count

        self._batch_mean = _batch_mean
        self._batch_var = _batch_var
        self._batch_count = _batch_count

    
        def update_from_moments(batch_mean, batch_var, batch_count):               
            for op in update_ops:
                sess.run(update_ops, feed_dict={
                    _batch_mean: batch_mean, 
                    _batch_var: batch_var,
                    _batch_count: batch_count
                })
                

        sess.run(tf.variables_initializer([_mean, _var, _count]))
        self.sess = sess
        self.update_from_moments = update_from_moments

    @property
    def mean(self):
        return self.sess.run(self._mean)

    @property
    def var(self):
        return self.sess.run(self._var)

    @property
    def count(self):
        return self.sess.run(self._count)

         
    def update(self, x):
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]
        self.update_from_moments(batch_mean, batch_var, batch_count)

          

def test_runningmeanstd():
    for (x1, x2, x3) in [
        (np.random.randn(3), np.random.randn(4), np.random.randn(5)),
        (np.random.randn(3,2), np.random.randn(4,2), np.random.randn(5,2)),
        ]:

        rms = RunningMeanStd(epsilon=0.0, shape=x1.shape[1:])

        x = np.concatenate([x1, x2, x3], axis=0)
        ms1 = [x.mean(axis=0), x.var(axis=0)]
        rms.update(x1)
        rms.update(x2)
        rms.update(x3)
        ms2 = [rms.mean, rms.var]

        np.testing.assert_allclose(ms1, ms2)

def test_tf_runningmeanstd():
    for (x1, x2, x3) in [
        (np.random.randn(3), np.random.randn(4), np.random.randn(5)),
        (np.random.randn(3,2), np.random.randn(4,2), np.random.randn(5,2)),
        ]:

        rms = TfRunningMeanStd(epsilon=0.0, shape=x1.shape[1:], scope='running_mean_std' + str(np.random.randint(0, 128)))

        x = np.concatenate([x1, x2, x3], axis=0)
        ms1 = [x.mean(axis=0), x.var(axis=0)]
        rms.update(x1)
        rms.update(x2)
        rms.update(x3)
        ms2 = [rms.mean, rms.var]

        np.testing.assert_allclose(ms1, ms2)
