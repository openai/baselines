try:
    from mpi4py import MPI
except ImportError:
    MPI = None

import tensorflow as tf, numpy as np

class RunningMeanStd(tf.Module):
    # https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm
    def __init__(self, epsilon=1e-2, shape=(), default_clip_range=np.inf):

        self._sum = tf.Variable(
            initial_value=np.zeros(shape=shape, dtype=np.float64),
            dtype=tf.float64,
            name="runningsum", trainable=False)
        self._sumsq = tf.Variable(
            initial_value=np.full(shape=shape, fill_value=epsilon, dtype=np.float64),
            dtype=tf.float64,
            name="runningsumsq", trainable=False)
        self._count = tf.Variable(
            initial_value=epsilon,
            dtype=tf.float64,
            name="count", trainable=False)
        self.shape = shape
        self.epsilon = epsilon
        self.default_clip_range = default_clip_range

    def update(self, x):
        x = x.astype('float64')
        n = int(np.prod(self.shape))
        addvec = np.concatenate([x.sum(axis=0).ravel(), np.square(x).sum(axis=0).ravel(), np.array([len(x)],dtype='float64')])
        totalvec = np.zeros(n*2+1, 'float64')
        if MPI is not None:
            # totalvec = np.zeros(n*2+1, 'float64')
            MPI.COMM_WORLD.Allreduce(addvec, totalvec, op=MPI.SUM)
        # else:
        #     totalvec = addvec
        self._sum.assign_add(totalvec[0:n].reshape(self.shape))
        self._sumsq.assign_add(totalvec[n:2*n].reshape(self.shape))
        self._count.assign_add(totalvec[2*n])

    @property
    def mean(self):
        return tf.cast(self._sum / self._count, tf.float32)

    @property
    def std(self):
        return tf.sqrt(tf.maximum(tf.cast(self._sumsq / self._count, tf.float32) - tf.square(self.mean), self.epsilon))

    def normalize(self, v, clip_range=None):
        if clip_range is None:
            clip_range = self.default_clip_range
        return tf.clip_by_value((v - self.mean) / self.std, -clip_range, clip_range)

    def denormalize(self, v):
        return self.mean + v * self.std