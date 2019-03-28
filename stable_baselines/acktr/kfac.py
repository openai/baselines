import re
from functools import reduce

import tensorflow as tf
import numpy as np

from stable_baselines.acktr.kfac_utils import detect_min_val, factor_reshape, gmatmul

KFAC_OPS = ['MatMul', 'Conv2D', 'BiasAdd']
KFAC_DEBUG = False


class KfacOptimizer:
    def __init__(self, learning_rate=0.01, momentum=0.9, clip_kl=0.01, kfac_update=2, stats_accum_iter=60,
                 full_stats_init=False, cold_iter=100, cold_lr=None, async_eigen_decomp=False,
                 async_stats=False, epsilon=1e-2, stats_decay=0.95, blockdiag_bias=False,
                 channel_fac=False, factored_damping=False, approx_t2=False,
                 use_float64=False, weight_decay_dict=None, max_grad_norm=0.5, verbose=1):
        """
        Kfac Optimizer for ACKTR models
        link: https://arxiv.org/pdf/1708.05144.pdf

        :param learning_rate: (float) The learning rate
        :param momentum: (float) The momentum value for the TensorFlow momentum optimizer
        :param clip_kl: (float) gradient clipping for Kullback leiber
        :param kfac_update: (int) update kfac after kfac_update steps
        :param stats_accum_iter: (int) how may steps to accumulate stats
        :param full_stats_init: (bool) whether or not to fully initalize stats
        :param cold_iter: (int) Cold start learning rate for how many steps
        :param cold_lr: (float) Cold start learning rate
        :param async_eigen_decomp: (bool) Use async eigen decomposition
        :param async_stats: (bool) Asynchronous stats update
        :param epsilon: (float) epsilon value for small numbers
        :param stats_decay: (float) the stats decay rate
        :param blockdiag_bias: (bool)
        :param channel_fac: (bool) factorization along the channels
        :param factored_damping: (bool) use factored damping
        :param approx_t2: (bool) approximate T2 act and grad fisher
        :param use_float64: (bool) use 64-bit float
        :param weight_decay_dict: (dict) custom weight decay coeff for a given gradient
        :param max_grad_norm: (float) The maximum value for the gradient clipping
        :param verbose: (int) verbosity level
        """
        self.max_grad_norm = max_grad_norm
        self._lr = learning_rate
        self._momentum = momentum
        self._clip_kl = clip_kl
        self._channel_fac = channel_fac
        self._kfac_update = kfac_update
        self._async_eigen_decomp = async_eigen_decomp
        self._async_stats = async_stats
        self._epsilon = epsilon
        self._stats_decay = stats_decay
        self._blockdiag_bias = blockdiag_bias
        self._approx_t2 = approx_t2
        self._use_float64 = use_float64
        self._factored_damping = factored_damping
        self._cold_iter = cold_iter
        self.verbose = verbose
        if cold_lr is None:
            # good heuristics
            self._cold_lr = self._lr  # * 3.
        else:
            self._cold_lr = cold_lr
        self._stats_accum_iter = stats_accum_iter
        if weight_decay_dict is None:
            weight_decay_dict = {}
        self._weight_decay_dict = weight_decay_dict
        self._diag_init_coeff = 0.
        self._full_stats_init = full_stats_init
        if not self._full_stats_init:
            self._stats_accum_iter = self._cold_iter

        self.sgd_step = tf.Variable(0, name='KFAC/sgd_step', trainable=False)
        self.global_step = tf.Variable(
            0, name='KFAC/global_step', trainable=False)
        self.cold_step = tf.Variable(0, name='KFAC/cold_step', trainable=False)
        self.factor_step = tf.Variable(
            0, name='KFAC/factor_step', trainable=False)
        self.stats_step = tf.Variable(
            0, name='KFAC/stats_step', trainable=False)
        self.v_f_v = tf.Variable(0., name='KFAC/vFv', trainable=False)

        self.factors = {}
        self.param_vars = []
        self.stats = {}
        self.stats_eigen = {}

    def get_factors(self, gradients, varlist):
        """
        get factors to update

        :param gradients: ([TensorFlow Tensor]) The gradients
        :param varlist: ([TensorFlow Tensor]) The parameters
        :return: ([TensorFlow Tensor]) The factors to update
        """
        default_graph = tf.get_default_graph()
        factor_tensors = {}
        fprop_tensors = []
        bprop_tensors = []
        op_types = []

        def _search_factors(gradient, graph):
            # hard coded search stratergy
            bprop_op = gradient.op
            bprop_op_name = bprop_op.name

            b_tensors = []
            f_tensors = []

            # combining additive gradient, assume they are the same op type and
            # indepedent
            if 'AddN' in bprop_op_name:
                factors = []
                for grad in gradient.op.inputs:
                    factors.append(_search_factors(grad, graph))
                op_names = [_item['opName'] for _item in factors]
                if self.verbose > 1:
                    # TODO: need to check all the attribute of the ops as well
                    print(gradient.name)
                    print(op_names)
                    print(len(np.unique(op_names)))
                assert len(np.unique(op_names)) == 1, \
                    'Error: {} is shared among different computation OPs'.format(gradient.name)

                b_tensors = reduce(lambda x, y: x + y,
                                   [_item['bpropFactors'] for _item in factors])
                if len(factors[0]['fpropFactors']) > 0:
                    f_tensors = reduce(
                        lambda x, y: x + y, [_item['fpropFactors'] for _item in factors])
                fprop_op_name = op_names[0]
                fprop_op = factors[0]['op']
            else:
                fprop_op_name = re.search('gradientsSampled(_[0-9]+|)/(.+?)_grad', bprop_op_name).group(2)
                fprop_op = graph.get_operation_by_name(fprop_op_name)
                if fprop_op.op_def.name in KFAC_OPS:
                    # Known OPs
                    b_tensor = [_i for _i in bprop_op.inputs if 'gradientsSampled' in _i.name][-1]
                    b_tensor_shape = fprop_op.outputs[0].get_shape()
                    if b_tensor.get_shape()[0].value is None:
                        b_tensor.set_shape(b_tensor_shape)
                    b_tensors.append(b_tensor)

                    if fprop_op.op_def.name == 'BiasAdd':
                        f_tensors = []
                    else:
                        f_tensors.append([_i for _i in fprop_op.inputs if param.op.name not in _i.name][0])
                    fprop_op_name = fprop_op.op_def.name
                else:
                    # unknown OPs, block approximation used
                    b_inputs_list = [_i for _i in bprop_op.inputs[0].op.inputs
                                     if 'gradientsSampled' in _i.name if 'Shape' not in _i.name]
                    if len(b_inputs_list) > 0:
                        b_tensor = b_inputs_list[0]
                        # only if tensor shape is defined, usually this will prevent tensor like Sum:0 to be used.
                        if b_tensor.get_shape():
                            b_tensor_shape = fprop_op.outputs[0].get_shape()
                            if len(b_tensor.get_shape()) > 0 and b_tensor.get_shape()[0].value is None:
                                b_tensor.set_shape(b_tensor_shape)
                            b_tensors.append(b_tensor)
                    fprop_op_name = 'UNK-' + fprop_op.op_def.name
                    op_types.append(fprop_op_name)

            return {'opName': fprop_op_name, 'op': fprop_op, 'fpropFactors': f_tensors, 'bpropFactors': b_tensors}

        for _grad, param in zip(gradients, varlist):
            if KFAC_DEBUG:
                print(('get factor for ' + param.name))
            found_factors = _search_factors(_grad, default_graph)
            factor_tensors[param] = found_factors

        # check associated weights and bias for homogeneous coordinate representation
        # and check redundent factors
        # TODO: there may be a bug to detect associate bias and weights for forking layer, e.g. in inception models.
        for param in varlist:
            factor_tensors[param]['assnWeights'] = None
            factor_tensors[param]['assnBias'] = None
        for param in varlist:
            if factor_tensors[param]['opName'] == 'BiasAdd':
                factor_tensors[param]['assnWeights'] = None
                for item in varlist:
                    if len(factor_tensors[item]['bpropFactors']) > 0:
                        if (set(factor_tensors[item]['bpropFactors']) == set(factor_tensors[param]['bpropFactors'])) \
                                and (len(factor_tensors[item]['fpropFactors']) > 0):
                            factor_tensors[param]['assnWeights'] = item
                            factor_tensors[item]['assnBias'] = param
                            factor_tensors[param]['bpropFactors'] = factor_tensors[
                                item]['bpropFactors']

        # concatenate the additive gradients along the batch dimension, i.e. assuming independence structure
        for key in ['fpropFactors', 'bpropFactors']:
            for i, param in enumerate(varlist):
                if len(factor_tensors[param][key]) > 0:
                    if (key + '_concat') not in factor_tensors[param]:
                        name_scope = factor_tensors[param][key][0].name.split(':')[
                            0]
                        with tf.name_scope(name_scope):
                            factor_tensors[param][
                                key + '_concat'] = tf.concat(factor_tensors[param][key], 0)
                else:
                    factor_tensors[param][key + '_concat'] = None
                for _, param2 in enumerate(varlist[(i + 1):]):
                    if (len(factor_tensors[param][key]) > 0) and (
                            set(factor_tensors[param2][key]) == set(factor_tensors[param][key])):
                        factor_tensors[param2][key] = factor_tensors[param][key]
                        factor_tensors[param2][
                            key + '_concat'] = factor_tensors[param][key + '_concat']

        if KFAC_DEBUG:
            for items in zip(varlist, fprop_tensors, bprop_tensors, op_types):
                print((items[0].name, factor_tensors[item]))
        self.factors = factor_tensors
        return factor_tensors

    def get_stats(self, factors, varlist):
        """
        return the stats values from the factors to update and the parameters

        :param factors: ([TensorFlow Tensor]) The factors to update
        :param varlist: ([TensorFlow Tensor]) The parameters
        :return: ([TensorFlow Tensor]) The stats values
        """
        if len(self.stats) == 0:
            # initialize stats variables on CPU because eigen decomp is
            # computed on CPU
            with tf.device('/cpu'):
                tmp_stats_cache = {}

                # search for tensor factors and
                # use block diag approx for the bias units
                for var in varlist:
                    bprop_factor = factors[var]['bpropFactors_concat']
                    op_type = factors[var]['opName']
                    if op_type == 'Conv2D':
                        operator_height = bprop_factor.get_shape()[1]
                        operator_width = bprop_factor.get_shape()[2]
                        if operator_height == 1 and operator_width == 1 and self._channel_fac:
                            # factorization along the channels do not support
                            # homogeneous coordinate
                            var_assn_bias = factors[var]['assnBias']
                            if var_assn_bias:
                                factors[var]['assnBias'] = None
                                factors[var_assn_bias]['assnWeights'] = None

                for var in varlist:
                    fprop_factor = factors[var]['fpropFactors_concat']
                    bprop_factor = factors[var]['bpropFactors_concat']
                    op_type = factors[var]['opName']
                    self.stats[var] = {'opName': op_type,
                                       'fprop_concat_stats': [],
                                       'bprop_concat_stats': [],
                                       'assnWeights': factors[var]['assnWeights'],
                                       'assnBias': factors[var]['assnBias'],
                                       }
                    if fprop_factor is not None:
                        if fprop_factor not in tmp_stats_cache:
                            if op_type == 'Conv2D':
                                kernel_height = var.get_shape()[0]
                                kernel_width = var.get_shape()[1]
                                n_channels = fprop_factor.get_shape()[-1]

                                operator_height = bprop_factor.get_shape()[1]
                                operator_width = bprop_factor.get_shape()[2]
                                if operator_height == 1 and operator_width == 1 and self._channel_fac:
                                    # factorization along the channels
                                    # assume independence between input channels and spatial
                                    # 2K-1 x 2K-1 covariance matrix and C x C covariance matrix
                                    # factorization along the channels do not
                                    # support homogeneous coordinate, assnBias
                                    # is always None
                                    fprop_factor2_size = kernel_height * kernel_width
                                    slot_fprop_factor_stats2 = tf.Variable(tf.diag(tf.ones(
                                        [fprop_factor2_size])) * self._diag_init_coeff,
                                                                           name='KFAC_STATS/' + fprop_factor.op.name,
                                                                           trainable=False)
                                    self.stats[var]['fprop_concat_stats'].append(
                                        slot_fprop_factor_stats2)

                                    fprop_factor_size = n_channels
                                else:
                                    # 2K-1 x 2K-1 x C x C covariance matrix
                                    # assume BHWC
                                    fprop_factor_size = kernel_height * kernel_width * n_channels
                            else:
                                # D x D covariance matrix
                                fprop_factor_size = fprop_factor.get_shape()[-1]

                            # use homogeneous coordinate
                            if not self._blockdiag_bias and self.stats[var]['assnBias']:
                                fprop_factor_size += 1

                            slot_fprop_factor_stats = tf.Variable(
                                tf.diag(tf.ones([fprop_factor_size])) * self._diag_init_coeff,
                                name='KFAC_STATS/' + fprop_factor.op.name, trainable=False)
                            self.stats[var]['fprop_concat_stats'].append(
                                slot_fprop_factor_stats)
                            if op_type != 'Conv2D':
                                tmp_stats_cache[fprop_factor] = self.stats[
                                    var]['fprop_concat_stats']
                        else:
                            self.stats[var][
                                'fprop_concat_stats'] = tmp_stats_cache[fprop_factor]

                    if bprop_factor is not None:
                        # no need to collect backward stats for bias vectors if
                        # using homogeneous coordinates
                        if not ((not self._blockdiag_bias) and self.stats[var]['assnWeights']):
                            if bprop_factor not in tmp_stats_cache:
                                slot_bprop_factor_stats = tf.Variable(tf.diag(tf.ones([bprop_factor.get_shape(
                                )[-1]])) * self._diag_init_coeff, name='KFAC_STATS/' + bprop_factor.op.name,
                                                                      trainable=False)
                                self.stats[var]['bprop_concat_stats'].append(
                                    slot_bprop_factor_stats)
                                tmp_stats_cache[bprop_factor] = self.stats[
                                    var]['bprop_concat_stats']
                            else:
                                self.stats[var][
                                    'bprop_concat_stats'] = tmp_stats_cache[bprop_factor]

        return self.stats

    def compute_and_apply_stats(self, loss_sampled, var_list=None):
        """
        compute and apply stats

        :param loss_sampled: ([TensorFlow Tensor]) the loss function output
        :param var_list: ([TensorFlow Tensor]) The parameters
        :return: (function) apply stats
        """
        varlist = var_list
        if varlist is None:
            varlist = tf.trainable_variables()

        stats = self.compute_stats(loss_sampled, var_list=varlist)
        return self.apply_stats(stats)

    def compute_stats(self, loss_sampled, var_list=None):
        """
        compute the stats values

        :param loss_sampled: ([TensorFlow Tensor]) the loss function output
        :param var_list: ([TensorFlow Tensor]) The parameters
        :return: ([TensorFlow Tensor]) stats updates
        """
        varlist = var_list
        if varlist is None:
            varlist = tf.trainable_variables()

        gradient_sampled = tf.gradients(loss_sampled, varlist, name='gradientsSampled')
        self.gradient_sampled = gradient_sampled

        # remove unused variables
        gradient_sampled, varlist = zip(*[(grad, var) for (grad, var) in zip(gradient_sampled, varlist)
                                          if grad is not None])

        factors = self.get_factors(gradient_sampled, varlist)
        stats = self.get_stats(factors, varlist)

        update_ops = []
        stats_updates = {}
        stats_updates_cache = {}
        for var in varlist:
            op_type = factors[var]['opName']
            fops = factors[var]['op']
            fprop_factor = factors[var]['fpropFactors_concat']
            fprop_stats_vars = stats[var]['fprop_concat_stats']
            bprop_factor = factors[var]['bpropFactors_concat']
            bprop_stats_vars = stats[var]['bprop_concat_stats']
            svd_factors = {}
            for stats_var in fprop_stats_vars:
                stats_var_dim = int(stats_var.get_shape()[0])
                if stats_var not in stats_updates_cache:
                    batch_size = (tf.shape(fprop_factor)[0])  # batch size
                    if op_type == 'Conv2D':
                        strides = fops.get_attr("strides")
                        padding = fops.get_attr("padding")
                        convkernel_size = var.get_shape()[0:3]

                        kernel_height = int(convkernel_size[0])
                        kernel_width = int(convkernel_size[1])
                        chan = int(convkernel_size[2])
                        flatten_size = int(kernel_height * kernel_width * chan)

                        operator_height = int(bprop_factor.get_shape()[1])
                        operator_width = int(bprop_factor.get_shape()[2])

                        if operator_height == 1 and operator_width == 1 and self._channel_fac:
                            # factorization along the channels
                            # assume independence among input channels
                            # factor = B x 1 x 1 x (KH xKW x C)
                            # patches = B x Oh x Ow x (KH xKW x C)
                            if len(svd_factors) == 0:
                                if KFAC_DEBUG:
                                    print(('approx %s act factor with rank-1 SVD factors' % var.name))
                                # find closest rank-1 approx to the feature map
                                S, U, V = tf.batch_svd(tf.reshape(
                                    fprop_factor, [-1, kernel_height * kernel_width, chan]))
                                # get rank-1 approx slides
                                sqrt_s1 = tf.expand_dims(tf.sqrt(S[:, 0, 0]), 1)
                                patches_k = U[:, :, 0] * sqrt_s1  # B x KH*KW
                                full_factor_shape = fprop_factor.get_shape()
                                patches_k.set_shape(
                                    [full_factor_shape[0], kernel_height * kernel_width])
                                patches_c = V[:, :, 0] * sqrt_s1  # B x C
                                patches_c.set_shape([full_factor_shape[0], chan])
                                svd_factors[chan] = patches_c
                                svd_factors[kernel_height * kernel_width] = patches_k
                            fprop_factor = svd_factors[stats_var_dim]

                        else:
                            # poor mem usage implementation
                            patches = tf.extract_image_patches(fprop_factor, ksizes=[1, convkernel_size[
                                0], convkernel_size[1], 1], strides=strides, rates=[1, 1, 1, 1], padding=padding)

                            if self._approx_t2:
                                if KFAC_DEBUG:
                                    print(('approxT2 act fisher for %s' % var.name))
                                # T^2 terms * 1/T^2, size: B x C
                                fprop_factor = tf.reduce_mean(patches, [1, 2])
                            else:
                                # size: (B x Oh x Ow) x C
                                fprop_factor = tf.reshape(
                                    patches, [-1, flatten_size]) / operator_height / operator_width
                    fprop_factor_size = int(fprop_factor.get_shape()[-1])
                    if stats_var_dim == (fprop_factor_size + 1) and not self._blockdiag_bias:
                        if op_type == 'Conv2D' and not self._approx_t2:
                            # correct padding for numerical stability (we
                            # divided out OhxOw from activations for T1 approx)
                            fprop_factor = tf.concat([fprop_factor, tf.ones(
                                [tf.shape(fprop_factor)[0], 1]) / operator_height / operator_width], 1)
                        else:
                            # use homogeneous coordinates
                            fprop_factor = tf.concat(
                                [fprop_factor, tf.ones([tf.shape(fprop_factor)[0], 1])], 1)

                    # average over the number of data points in a batch
                    # divided by B
                    cov = tf.matmul(fprop_factor, fprop_factor,
                                    transpose_a=True) / tf.cast(batch_size, tf.float32)
                    update_ops.append(cov)
                    stats_updates[stats_var] = cov
                    if op_type != 'Conv2D':
                        # HACK: for convolution we recompute fprop stats for
                        # every layer including forking layers
                        stats_updates_cache[stats_var] = cov

            for stats_var in bprop_stats_vars:
                if stats_var not in stats_updates_cache:
                    bprop_factor_shape = bprop_factor.get_shape()
                    batch_size = tf.shape(bprop_factor)[0]  # batch size
                    chan = int(bprop_factor_shape[-1])  # num channels
                    if op_type == 'Conv2D' or len(bprop_factor_shape) == 4:
                        if fprop_factor is not None:
                            if self._approx_t2:
                                if KFAC_DEBUG:
                                    print(('approxT2 grad fisher for %s' % var.name))
                                bprop_factor = tf.reduce_sum(
                                    bprop_factor, [1, 2])  # T^2 terms * 1/T^2
                            else:
                                bprop_factor = tf.reshape(
                                    bprop_factor, [-1, chan]) * operator_height * operator_width  # T * 1/T terms
                        else:
                            # just doing block diag approx. spatial independent
                            # structure does not apply here. summing over
                            # spatial locations
                            if KFAC_DEBUG:
                                print(('block diag approx fisher for %s' % var.name))
                            bprop_factor = tf.reduce_sum(bprop_factor, [1, 2])

                    # assume sampled loss is averaged. TODO:figure out better
                    # way to handle this
                    bprop_factor *= tf.cast(batch_size, tf.float32)
                    ##

                    cov_b = tf.matmul(bprop_factor, bprop_factor,
                                      transpose_a=True) / tf.cast(tf.shape(bprop_factor)[0], tf.float32)

                    update_ops.append(cov_b)
                    stats_updates[stats_var] = cov_b
                    stats_updates_cache[stats_var] = cov_b

        if KFAC_DEBUG:
            a_key = list(stats_updates.keys())[0]
            stats_updates[a_key] = tf.Print(stats_updates[a_key], [tf.convert_to_tensor('step:'), self.global_step,
                                                                   tf.convert_to_tensor('computing stats')])
        self.stats_updates = stats_updates
        return stats_updates

    def apply_stats(self, stats_updates):
        """
        compute stats and update/apply the new stats to the running average

        :param stats_updates: ([TensorFlow Tensor]) The stats updates
        :return: (function) update stats operation
        """

        def _update_accum_stats():
            if self._full_stats_init:
                return tf.cond(tf.greater(self.sgd_step, self._cold_iter), lambda: tf.group(
                    *self._apply_stats(stats_updates, accumulate=True, accumulate_coeff=1. / self._stats_accum_iter)),
                               tf.no_op)
            else:
                return tf.group(
                    *self._apply_stats(stats_updates, accumulate=True, accumulate_coeff=1. / self._stats_accum_iter))

        def _update_running_avg_stats(stats_updates):
            return tf.group(*self._apply_stats(stats_updates))

        if self._async_stats:
            # asynchronous stats update
            update_stats = self._apply_stats(stats_updates)

            queue = tf.FIFOQueue(1, [item.dtype for item in update_stats], shapes=[
                item.get_shape() for item in update_stats])
            enqueue_op = queue.enqueue(update_stats)

            def dequeue_stats_op():
                return queue.dequeue()

            self.qr_stats = tf.train.QueueRunner(queue, [enqueue_op])
            update_stats_op = tf.cond(tf.equal(queue.size(), tf.convert_to_tensor(
                0)), tf.no_op, lambda: tf.group(*[dequeue_stats_op(), ]))
        else:
            # synchronous stats update
            update_stats_op = tf.cond(tf.greater_equal(self.stats_step, self._stats_accum_iter),
                                      lambda: _update_running_avg_stats(stats_updates), _update_accum_stats)
        self._update_stats_op = update_stats_op
        return update_stats_op

    def _apply_stats(self, stats_updates, accumulate=False, accumulate_coeff=0.):
        update_ops = []
        # obtain the stats var list
        for stats_var in stats_updates:
            stats_new = stats_updates[stats_var]
            if accumulate:
                # simple superbatch averaging
                update_op = tf.assign_add(
                    stats_var, accumulate_coeff * stats_new, use_locking=True)
            else:
                # exponential running averaging
                update_op = tf.assign(
                    stats_var, stats_var * self._stats_decay, use_locking=True)
                update_op = tf.assign_add(
                    update_op, (1. - self._stats_decay) * stats_new, use_locking=True)
            update_ops.append(update_op)

        with tf.control_dependencies(update_ops):
            stats_step_op = tf.assign_add(self.stats_step, 1)

        if KFAC_DEBUG:
            stats_step_op = (tf.Print(stats_step_op,
                                      [tf.convert_to_tensor('step:'),
                                       self.global_step,
                                       tf.convert_to_tensor('fac step:'),
                                       self.factor_step,
                                       tf.convert_to_tensor('sgd step:'),
                                       self.sgd_step,
                                       tf.convert_to_tensor('Accum:'),
                                       tf.convert_to_tensor(accumulate),
                                       tf.convert_to_tensor('Accum coeff:'),
                                       tf.convert_to_tensor(accumulate_coeff),
                                       tf.convert_to_tensor('stat step:'),
                                       self.stats_step, update_ops[0], update_ops[1]]))
        return [stats_step_op, ]

    def get_stats_eigen(self, stats=None):
        """
        Return the eigen values from the stats

        :param stats: ([TensorFlow Tensor]) The stats
        :return: ([TensorFlow Tensor]) The stats eigen values
        """
        if len(self.stats_eigen) == 0:
            stats_eigen = {}
            if stats is None:
                stats = self.stats

            tmp_eigen_cache = {}
            with tf.device('/cpu:0'):
                for var in stats:
                    for key in ['fprop_concat_stats', 'bprop_concat_stats']:
                        for stats_var in stats[var][key]:
                            if stats_var not in tmp_eigen_cache:
                                stats_dim = stats_var.get_shape()[1].value
                                eigen_values = tf.Variable(tf.ones(
                                    [stats_dim]), name='KFAC_FAC/' + stats_var.name.split(':')[0] + '/e',
                                    trainable=False)
                                eigen_vectors = tf.Variable(tf.diag(tf.ones(
                                    [stats_dim])), name='KFAC_FAC/' + stats_var.name.split(':')[0] + '/Q',
                                    trainable=False)
                                stats_eigen[stats_var] = {'e': eigen_values, 'Q': eigen_vectors}
                                tmp_eigen_cache[
                                    stats_var] = stats_eigen[stats_var]
                            else:
                                stats_eigen[stats_var] = tmp_eigen_cache[
                                    stats_var]
            self.stats_eigen = stats_eigen
        return self.stats_eigen

    def compute_stats_eigen(self):
        """
        compute the eigen decomp using copied var stats to avoid concurrent read/write from other queue

        :return: ([TensorFlow Tensor]) update operations
        """
        # TODO: figure out why this op has delays (possibly moving eigenvectors around?)
        with tf.device('/cpu:0'):
            stats_eigen = self.stats_eigen
            computed_eigen = {}
            eigen_reverse_lookup = {}
            update_ops = []
            # sync copied stats
            with tf.control_dependencies([]):
                for stats_var in stats_eigen:
                    if stats_var not in computed_eigen:
                        eigen_decomposition = tf.self_adjoint_eig(stats_var)
                        eigen_values = eigen_decomposition[0]
                        eigen_vectors = eigen_decomposition[1]
                        if self._use_float64:
                            eigen_values = tf.cast(eigen_values, tf.float64)
                            eigen_vectors = tf.cast(eigen_vectors, tf.float64)
                        update_ops.append(eigen_values)
                        update_ops.append(eigen_vectors)
                        computed_eigen[stats_var] = {'e': eigen_values, 'Q': eigen_vectors}
                        eigen_reverse_lookup[eigen_values] = stats_eigen[stats_var]['e']
                        eigen_reverse_lookup[eigen_vectors] = stats_eigen[stats_var]['Q']

            self.eigen_reverse_lookup = eigen_reverse_lookup
            self.eigen_update_list = update_ops

            if KFAC_DEBUG:
                self.eigen_update_list = [item for item in update_ops]
                with tf.control_dependencies(update_ops):
                    update_ops.append(tf.Print(tf.constant(
                        0.), [tf.convert_to_tensor('computed factor eigen')]))

        return update_ops

    def apply_stats_eigen(self, eigen_list):
        """
        apply the update using the eigen values of the stats

        :param eigen_list: ([TensorFlow Tensor]) The list of eigen values of the stats
        :return: ([TensorFlow Tensor]) update operations
        """
        update_ops = []
        if self.verbose > 1:
            print(('updating %d eigenvalue/vectors' % len(eigen_list)))
        for _, (tensor, mark) in enumerate(zip(eigen_list, self.eigen_update_list)):
            stats_eigen_var = self.eigen_reverse_lookup[mark]
            update_ops.append(
                tf.assign(stats_eigen_var, tensor, use_locking=True))

        with tf.control_dependencies(update_ops):
            factor_step_op = tf.assign_add(self.factor_step, 1)
            update_ops.append(factor_step_op)
            if KFAC_DEBUG:
                update_ops.append(tf.Print(tf.constant(
                    0.), [tf.convert_to_tensor('updated kfac factors')]))
        return update_ops

    def get_kfac_precond_updates(self, gradlist, varlist):
        """
        return the KFAC updates

        :param gradlist: ([TensorFlow Tensor]) The gradients
        :param varlist: ([TensorFlow Tensor]) The parameters
        :return: ([TensorFlow Tensor]) the update list
        """
        v_g = 0.

        assert len(self.stats) > 0
        assert len(self.stats_eigen) > 0
        assert len(self.factors) > 0
        counter = 0

        grad_dict = {var: grad for grad, var in zip(gradlist, varlist)}

        for grad, var in zip(gradlist, varlist):
            grad_reshape = False

            fprop_factored_fishers = self.stats[var]['fprop_concat_stats']
            bprop_factored_fishers = self.stats[var]['bprop_concat_stats']

            if (len(fprop_factored_fishers) + len(bprop_factored_fishers)) > 0:
                counter += 1
                grad_shape = grad.get_shape()
                if len(grad.get_shape()) > 2:
                    # reshape conv kernel parameters
                    kernel_width = int(grad.get_shape()[0])
                    kernel_height = int(grad.get_shape()[1])
                    n_channels = int(grad.get_shape()[2])
                    depth = int(grad.get_shape()[3])

                    if len(fprop_factored_fishers) > 1 and self._channel_fac:
                        # reshape conv kernel parameters into tensor
                        grad = tf.reshape(grad, [kernel_width * kernel_height, n_channels, depth])
                    else:
                        # reshape conv kernel parameters into 2D grad
                        grad = tf.reshape(grad, [-1, depth])
                    grad_reshape = True
                elif len(grad.get_shape()) == 1:
                    # reshape bias or 1D parameters

                    grad = tf.expand_dims(grad, 0)
                    grad_reshape = True

                if (self.stats[var]['assnBias'] is not None) and not self._blockdiag_bias:
                    # use homogeneous coordinates only works for 2D grad.
                    # TODO: figure out how to factorize bias grad
                    # stack bias grad
                    var_assn_bias = self.stats[var]['assnBias']
                    grad = tf.concat(
                        [grad, tf.expand_dims(grad_dict[var_assn_bias], 0)], 0)

                # project gradient to eigen space and reshape the eigenvalues
                # for broadcasting
                eig_vals = []

                for idx, stats in enumerate(self.stats[var]['fprop_concat_stats']):
                    eigen_vectors = self.stats_eigen[stats]['Q']
                    eigen_values = detect_min_val(self.stats_eigen[stats][
                                                      'e'], var, name='act', debug=KFAC_DEBUG)

                    eigen_vectors, eigen_values = factor_reshape(eigen_vectors, eigen_values,
                                                                 grad, fac_idx=idx, f_type='act')
                    eig_vals.append(eigen_values)
                    grad = gmatmul(eigen_vectors, grad, transpose_a=True, reduce_dim=idx)

                for idx, stats in enumerate(self.stats[var]['bprop_concat_stats']):
                    eigen_vectors = self.stats_eigen[stats]['Q']
                    eigen_values = detect_min_val(self.stats_eigen[stats][
                                                      'e'], var, name='grad', debug=KFAC_DEBUG)

                    eigen_vectors, eigen_values = factor_reshape(eigen_vectors, eigen_values,
                                                                 grad, fac_idx=idx, f_type='grad')
                    eig_vals.append(eigen_values)
                    grad = gmatmul(grad, eigen_vectors, transpose_b=False, reduce_dim=idx)

                # whiten using eigenvalues
                weight_decay_coeff = 0.
                if var in self._weight_decay_dict:
                    weight_decay_coeff = self._weight_decay_dict[var]
                    if KFAC_DEBUG:
                        print(('weight decay coeff for %s is %f' % (var.name, weight_decay_coeff)))

                if self._factored_damping:
                    if KFAC_DEBUG:
                        print(('use factored damping for %s' % var.name))
                    coeffs = 1.
                    num_factors = len(eig_vals)
                    # compute the ratio of two trace norm of the left and right
                    # KFac matrices, and their generalization
                    if len(eig_vals) == 1:
                        damping = self._epsilon + weight_decay_coeff
                    else:
                        damping = tf.pow(
                            self._epsilon + weight_decay_coeff, 1. / num_factors)
                    eig_vals_tnorm_avg = [tf.reduce_mean(
                        tf.abs(e)) for e in eig_vals]
                    for eigen_val, e_tnorm in zip(eig_vals, eig_vals_tnorm_avg):
                        eig_tnorm_neg_list = [
                            item for item in eig_vals_tnorm_avg if item != e_tnorm]
                        if len(eig_vals) == 1:
                            adjustment = 1.
                        elif len(eig_vals) == 2:
                            adjustment = tf.sqrt(
                                e_tnorm / eig_tnorm_neg_list[0])
                        else:
                            eig_tnorm_neg_list_prod = reduce(
                                lambda x, y: x * y, eig_tnorm_neg_list)
                            adjustment = tf.pow(
                                tf.pow(e_tnorm, num_factors - 1.) / eig_tnorm_neg_list_prod, 1. / num_factors)
                        coeffs *= (eigen_val + adjustment * damping)
                else:
                    coeffs = 1.
                    damping = (self._epsilon + weight_decay_coeff)
                    for eigen_val in eig_vals:
                        coeffs *= eigen_val
                    coeffs += damping

                grad /= coeffs

                # project gradient back to euclidean space
                for idx, stats in enumerate(self.stats[var]['fprop_concat_stats']):
                    eigen_vectors = self.stats_eigen[stats]['Q']
                    grad = gmatmul(eigen_vectors, grad, transpose_a=False, reduce_dim=idx)

                for idx, stats in enumerate(self.stats[var]['bprop_concat_stats']):
                    eigen_vectors = self.stats_eigen[stats]['Q']
                    grad = gmatmul(grad, eigen_vectors, transpose_b=True, reduce_dim=idx)

                if (self.stats[var]['assnBias'] is not None) and not self._blockdiag_bias:
                    # use homogeneous coordinates only works for 2D grad.
                    # TODO: figure out how to factorize bias grad
                    # un-stack bias grad
                    var_assn_bias = self.stats[var]['assnBias']
                    c_plus_one = int(grad.get_shape()[0])
                    grad_assn_bias = tf.reshape(tf.slice(grad,
                                                         begin=[
                                                             c_plus_one - 1, 0],
                                                         size=[1, -1]), var_assn_bias.get_shape())
                    grad_assn_weights = tf.slice(grad,
                                                 begin=[0, 0],
                                                 size=[c_plus_one - 1, -1])
                    grad_dict[var_assn_bias] = grad_assn_bias
                    grad = grad_assn_weights

                if grad_reshape:
                    grad = tf.reshape(grad, grad_shape)

                grad_dict[var] = grad

        if self.verbose > 1:
            print(('projecting %d gradient matrices' % counter))

        for grad_1, var in zip(gradlist, varlist):
            grad = grad_dict[var]
            # clipping
            if KFAC_DEBUG:
                print(('apply clipping to %s' % var.name))
                tf.Print(grad, [tf.sqrt(tf.reduce_sum(tf.pow(grad, 2)))], "Euclidean norm of new grad")
            local_vg = tf.reduce_sum(grad * grad_1 * (self._lr * self._lr))
            v_g += local_vg

        # rescale everything
        if KFAC_DEBUG:
            print('apply vFv clipping')

        scaling = tf.minimum(1., tf.sqrt(self._clip_kl / v_g))
        if KFAC_DEBUG:
            scaling = tf.Print(scaling, [tf.convert_to_tensor(
                'clip: '), scaling, tf.convert_to_tensor(' vFv: '), v_g])
        with tf.control_dependencies([tf.assign(self.v_f_v, v_g)]):
            updatelist = [grad_dict[var] for var in varlist]
            for i, item in enumerate(updatelist):
                updatelist[i] = scaling * item

        return updatelist

    @classmethod
    def compute_gradients(cls, loss, var_list=None):
        """
        compute the gradients from the loss and the parameters

        :param loss: ([TensorFlow Tensor]) The loss
        :param var_list: ([TensorFlow Tensor]) The parameters
        :return: ([TensorFlow Tensor]) the gradient
        """
        varlist = var_list
        if varlist is None:
            varlist = tf.trainable_variables()
        gradients = tf.gradients(loss, varlist)

        return [(a, b) for a, b in zip(gradients, varlist)]

    def apply_gradients_kfac(self, grads):
        """
        apply the kfac gradient

        :param grads: ([TensorFlow Tensor]) the gradient
        :return: ([function], QueueRunner) Update functions, queue operation runner
        """
        grad, varlist = list(zip(*grads))

        if len(self.stats_eigen) == 0:
            self.get_stats_eigen()

        queue_runner = None
        # launch eigen-decomp on a queue thread
        if self._async_eigen_decomp:
            if self.verbose > 1:
                print('Using async eigen decomposition')
            # get a list of factor loading tensors
            factor_ops_dummy = self.compute_stats_eigen()

            # define a queue for the list of factor loading tensors
            queue = tf.FIFOQueue(1, [item.dtype for item in factor_ops_dummy],
                                 shapes=[item.get_shape() for item in factor_ops_dummy])
            enqueue_op = tf.cond(
                tf.logical_and(tf.equal(tf.mod(self.stats_step, self._kfac_update), tf.convert_to_tensor(
                    0)), tf.greater_equal(self.stats_step, self._stats_accum_iter)),
                lambda: queue.enqueue(self.compute_stats_eigen()), tf.no_op)

            def dequeue_op():
                return queue.dequeue()

            queue_runner = tf.train.QueueRunner(queue, [enqueue_op])

        update_ops = []
        global_step_op = tf.assign_add(self.global_step, 1)
        update_ops.append(global_step_op)

        with tf.control_dependencies([global_step_op]):

            # compute updates
            assert self._update_stats_op is not None
            update_ops.append(self._update_stats_op)
            dependency_list = []
            if not self._async_eigen_decomp:
                dependency_list.append(self._update_stats_op)

            with tf.control_dependencies(dependency_list):
                def no_op_wrapper():
                    return tf.group(*[tf.assign_add(self.cold_step, 1)])

                if not self._async_eigen_decomp:
                    # synchronous eigen-decomp updates
                    update_factor_ops = tf.cond(tf.logical_and(tf.equal(tf.mod(self.stats_step, self._kfac_update),
                                                                        tf.convert_to_tensor(0)),
                                                               tf.greater_equal(self.stats_step,
                                                                                self._stats_accum_iter)),
                                                lambda: tf.group(*self.apply_stats_eigen(self.compute_stats_eigen())),
                                                no_op_wrapper)
                else:
                    # asynchronous eigen-decomp updates using queue
                    update_factor_ops = tf.cond(tf.greater_equal(self.stats_step, self._stats_accum_iter),
                                                lambda: tf.cond(tf.equal(queue.size(), tf.convert_to_tensor(0)),
                                                                tf.no_op,

                                                                lambda: tf.group(
                                                                    *self.apply_stats_eigen(dequeue_op())),
                                                                ),
                                                no_op_wrapper)

                update_ops.append(update_factor_ops)

                with tf.control_dependencies([update_factor_ops]):
                    def grad_op():
                        return list(grad)

                    def get_kfac_grad_op():
                        return self.get_kfac_precond_updates(grad, varlist)

                    u = tf.cond(tf.greater(self.factor_step,
                                           tf.convert_to_tensor(0)), get_kfac_grad_op, grad_op)

                    optim = tf.train.MomentumOptimizer(
                        self._lr * (1. - self._momentum), self._momentum)

                    # optim = tf.train.AdamOptimizer(self._lr, epsilon=0.01)

                    def optim_op():
                        def update_optim_op():
                            if self._full_stats_init:
                                return tf.cond(tf.greater(self.factor_step, tf.convert_to_tensor(0)),
                                               lambda: optim.apply_gradients(list(zip(u, varlist))), tf.no_op)
                            else:
                                return optim.apply_gradients(list(zip(u, varlist)))

                        if self._full_stats_init:
                            return tf.cond(tf.greater_equal(self.stats_step, self._stats_accum_iter), update_optim_op,
                                           tf.no_op)
                        else:
                            return tf.cond(tf.greater_equal(self.sgd_step, self._cold_iter), update_optim_op, tf.no_op)

                    update_ops.append(optim_op())

        return tf.group(*update_ops), queue_runner

    def apply_gradients(self, grads):
        """
        apply the gradient

        :param grads: ([TensorFlow Tensor]) the gradient
        :return: (function, QueueRunner) train operation, queue operation runner
        """
        cold_optim = tf.train.MomentumOptimizer(self._cold_lr, self._momentum)

        def _cold_sgd_start():
            sgd_grads, sgd_var = zip(*grads)

            if self.max_grad_norm is not None:
                sgd_grads, _ = tf.clip_by_global_norm(sgd_grads, self.max_grad_norm)

            sgd_grads = list(zip(sgd_grads, sgd_var))

            sgd_step_op = tf.assign_add(self.sgd_step, 1)
            cold_optim_op = cold_optim.apply_gradients(sgd_grads)
            if KFAC_DEBUG:
                with tf.control_dependencies([sgd_step_op, cold_optim_op]):
                    sgd_step_op = tf.Print(
                        sgd_step_op, [self.sgd_step, tf.convert_to_tensor('doing cold sgd step')])
            return tf.group(*[sgd_step_op, cold_optim_op])

        # remove unused variables
        grads = [(grad, var) for (grad, var) in grads if grad is not None]

        kfac_optim_op, queue_runner = self.apply_gradients_kfac(grads)

        def _warm_kfac_start():
            return kfac_optim_op

        return tf.cond(tf.greater(self.sgd_step, self._cold_iter), _warm_kfac_start, _cold_sgd_start), queue_runner

    def minimize(self, loss, loss_sampled, var_list=None):
        """
        minimize the gradient loss

        :param loss: ([TensorFlow Tensor]) The loss
        :param loss_sampled: ([TensorFlow Tensor]) the loss function output
        :param var_list: ([TensorFlow Tensor]) The parameters
        :return: (function, q_runner) train operation, queue operation runner
        """
        grads = self.compute_gradients(loss, var_list=var_list)
        self.compute_and_apply_stats(loss_sampled, var_list=var_list)
        return self.apply_gradients(grads)
