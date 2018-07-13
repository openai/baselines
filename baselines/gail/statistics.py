"""
This code is highly based on https://github.com/carpedm20/deep-rl-tensorflow/blob/master/agents/statistic.py
"""

import tensorflow as tf
import numpy as np

import baselines.common.tf_util as tf_util


class Stats:

    def __init__(self, scalar_keys=None, histogram_keys=None):
        """
        initialize the placeholders from the input keys, for summary logging

        :param scalar_keys: ([str]) the name of all the scalar inputs
        :param histogram_keys: ([str]) the name of all the histogram inputs
        """
        if scalar_keys is None:
            scalar_keys = []
        if histogram_keys is None:
            histogram_keys = []
        self.scalar_keys = scalar_keys
        self.histogram_keys = histogram_keys
        self.scalar_summaries = []
        self.scalar_summaries_ph = []
        self.histogram_summaries_ph = []
        self.histogram_summaries = []
        with tf.variable_scope('summary'):
            for k in scalar_keys:
                ph = tf.placeholder('float32', None, name=k+'.scalar.summary')
                sm = tf.summary.scalar(k+'.scalar.summary', ph)
                self.scalar_summaries_ph.append(ph)
                self.scalar_summaries.append(sm)
            for k in histogram_keys:
                ph = tf.placeholder('float32', None, name=k+'.histogram.summary')
                sm = tf.summary.scalar(k+'.histogram.summary', ph)
                self.histogram_summaries_ph.append(ph)
                self.histogram_summaries.append(sm)

        self.summaries = tf.summary.merge(self.scalar_summaries+self.histogram_summaries)

    def add_all_summary(self, writer, values, iter):
        """
        Note that the order of the incoming ```values``` should be the same as the that of the
                   ```scalar_keys``` given in ```__init__```

        :param writer: (TensorFlow FileWriter) the writer
        :param values: (TensorFlow Tensor or numpy Number) the input for the summary run
        :param iter: (Number) the global step value
        """
        if np.sum(np.isnan(values)+0) != 0:
            return
        sess = tf_util.get_session()
        keys = self.scalar_summaries_ph + self.histogram_summaries_ph
        feed_dict = {}
        for k, v in zip(keys, values):
            feed_dict.update({k: v})
        summaries_str = sess.run(self.summaries, feed_dict)
        writer.add_summary(summaries_str, iter)
