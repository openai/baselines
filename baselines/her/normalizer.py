import threading

import numpy as np
from mpi4py import MPI
import tensorflow as tf

from baselines.her.util import reshape_for_broadcasting


class Normalizer:
    def __init__(self, size, eps=1e-2, default_clip_range=np.inf, sess=None):
        """A normalizer that ensures that observations are approximately distributed according to
        a standard Normal distribution (i.e. have mean zero and variance one).

        Args:
            size (int): the size of the observation to be normalized
            eps (float): a small constant that avoids underflows
            default_clip_range (float): normalized observations are clipped to be in
                [-default_clip_range, default_clip_range]
            sess (object): the TensorFlow session to be used
        """
        self.size = size
        self.eps = eps
        self.default_clip_range = default_clip_range
        self.sess = sess if sess is not None else tf.get_default_session()

        self.local_sum = np.zeros(self.size, np.float32)
        self.local_sumsq = np.zeros(self.size, np.float32)
        self.local_count = np.zeros(1, np.float32)

        self.sum_tf = tf.get_variable(
            initializer=tf.zeros_initializer(), shape=self.local_sum.shape, name='sum',
            trainable=False, dtype=tf.float32)
        self.sumsq_tf = tf.get_variable(
            initializer=tf.zeros_initializer(), shape=self.local_sumsq.shape, name='sumsq',
            trainable=False, dtype=tf.float32)
        self.count_tf = tf.get_variable(
            initializer=tf.ones_initializer(), shape=self.local_count.shape, name='count',
            trainable=False, dtype=tf.float32)
        self.mean = tf.get_variable(
            initializer=tf.zeros_initializer(), shape=(self.size,), name='mean',
            trainable=False, dtype=tf.float32)
        self.std = tf.get_variable(
            initializer=tf.ones_initializer(), shape=(self.size,), name='std',
            trainable=False, dtype=tf.float32)
        self.count_pl = tf.placeholder(name='count_pl', shape=(1,), dtype=tf.float32)
        self.sum_pl = tf.placeholder(name='sum_pl', shape=(self.size,), dtype=tf.float32)
        self.sumsq_pl = tf.placeholder(name='sumsq_pl', shape=(self.size,), dtype=tf.float32)

        self.update_op = tf.group(
            self.count_tf.assign_add(self.count_pl),
            self.sum_tf.assign_add(self.sum_pl),
            self.sumsq_tf.assign_add(self.sumsq_pl)
        )
        self.recompute_op = tf.group(
            tf.assign(self.mean, self.sum_tf / self.count_tf),
            tf.assign(self.std, tf.sqrt(tf.maximum(
                tf.square(self.eps),
                self.sumsq_tf / self.count_tf - tf.square(self.sum_tf / self.count_tf)
            ))),
        )
        self.lock = threading.Lock()

    def update(self, v):
        v = v.reshape(-1, self.size)

        with self.lock:
            self.local_sum += v.sum(axis=0)
            self.local_sumsq += (np.square(v)).sum(axis=0)
            self.local_count[0] += v.shape[0]

    def normalize(self, v, clip_range=None):
        if clip_range is None:
            clip_range = self.default_clip_range
        mean = reshape_for_broadcasting(self.mean, v)
        std = reshape_for_broadcasting(self.std,  v)
        return tf.clip_by_value((v - mean) / std, -clip_range, clip_range)

    def denormalize(self, v):
        mean = reshape_for_broadcasting(self.mean, v)
        std = reshape_for_broadcasting(self.std,  v)
        return mean + v * std

    def _mpi_average(self, x):
        buf = np.zeros_like(x)
        MPI.COMM_WORLD.Allreduce(x, buf, op=MPI.SUM)
        buf /= MPI.COMM_WORLD.Get_size()
        return buf

    def synchronize(self, local_sum, local_sumsq, local_count, root=None):
        local_sum[...] = self._mpi_average(local_sum)
        local_sumsq[...] = self._mpi_average(local_sumsq)
        local_count[...] = self._mpi_average(local_count)
        return local_sum, local_sumsq, local_count

    def recompute_stats(self):
        with self.lock:
            # Copy over results.
            local_count = self.local_count.copy()
            local_sum = self.local_sum.copy()
            local_sumsq = self.local_sumsq.copy()

            # Reset.
            self.local_count[...] = 0
            self.local_sum[...] = 0
            self.local_sumsq[...] = 0

        # We perform the synchronization outside of the lock to keep the critical section as short
        # as possible.
        synced_sum, synced_sumsq, synced_count = self.synchronize(
            local_sum=local_sum, local_sumsq=local_sumsq, local_count=local_count)

        self.sess.run(self.update_op, feed_dict={
            self.count_pl: synced_count,
            self.sum_pl: synced_sum,
            self.sumsq_pl: synced_sumsq,
        })
        self.sess.run(self.recompute_op)


class IdentityNormalizer:
    def __init__(self, size, std=1.):
        self.size = size
        self.mean = tf.zeros(self.size, tf.float32)
        self.std = std * tf.ones(self.size, tf.float32)

    def update(self, x):
        pass

    def normalize(self, x, clip_range=None):
        return x / self.std

    def denormalize(self, x):
        return self.std * x

    def synchronize(self):
        pass

    def recompute_stats(self):
        pass
