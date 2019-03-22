import tensorflow as tf
import numpy as np
import re

 # flake8: noqa F403, F405
from baselines.acktr.kfac_utils import *
from functools import reduce

KFAC_OPS = ['MatMul', 'Conv2D', 'BiasAdd']
KFAC_DEBUG = False


class KfacOptimizer():
    # note that KfacOptimizer will be truly synchronous (and thus deterministic) only if a single-threaded session is used
    def __init__(self, learning_rate=0.01, momentum=0.9, clip_kl=0.01, kfac_update=2, stats_accum_iter=60, full_stats_init=False, cold_iter=100, cold_lr=None, is_async=False, async_stats=False, epsilon=1e-2, stats_decay=0.95, blockdiag_bias=False, channel_fac=False, factored_damping=False, approxT2=False, use_float64=False, weight_decay_dict={},max_grad_norm=0.5):
        self.max_grad_norm = max_grad_norm
        self._lr = learning_rate
        self._momentum = momentum
        self._clip_kl = clip_kl
        self._channel_fac = channel_fac
        self._kfac_update = kfac_update
        self._async = is_async
        self._async_stats = async_stats
        self._epsilon = epsilon
        self._stats_decay = stats_decay
        self._blockdiag_bias = blockdiag_bias
        self._approxT2 = approxT2
        self._use_float64 = use_float64
        self._factored_damping = factored_damping
        self._cold_iter = cold_iter
        if cold_lr == None:
            # good heuristics
            self._cold_lr = self._lr# * 3.
        else:
            self._cold_lr = cold_lr
        self._stats_accum_iter = stats_accum_iter
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
        self.vFv = tf.Variable(0., name='KFAC/vFv', trainable=False)

        self.factors = {}
        self.param_vars = []
        self.stats = {}
        self.stats_eigen = {}

    def getFactors(self, g, varlist):
        graph = tf.get_default_graph()
        factorTensors = {}
        fpropTensors = []
        bpropTensors = []
        opTypes = []
        fops = []

        def searchFactors(gradient, graph):
            # hard coded search stratergy
            bpropOp = gradient.op
            bpropOp_name = bpropOp.name

            bTensors = []
            fTensors = []

            # combining additive gradient, assume they are the same op type and
            # indepedent
            if 'AddN' in bpropOp_name:
                factors = []
                for g in gradient.op.inputs:
                    factors.append(searchFactors(g, graph))
                op_names = [item['opName'] for item in factors]
                # TO-DO: need to check all the attribute of the ops as well
                print (gradient.name)
                print (op_names)
                print (len(np.unique(op_names)))
                assert len(np.unique(op_names)) == 1, gradient.name + \
                    ' is shared among different computation OPs'

                bTensors = reduce(lambda x, y: x + y,
                                  [item['bpropFactors'] for item in factors])
                if len(factors[0]['fpropFactors']) > 0:
                    fTensors = reduce(
                        lambda x, y: x + y, [item['fpropFactors'] for item in factors])
                fpropOp_name = op_names[0]
                fpropOp = factors[0]['op']
            else:
                fpropOp_name = re.search(
                    'gradientsSampled(_[0-9]+|)/(.+?)_grad', bpropOp_name).group(2)
                fpropOp = graph.get_operation_by_name(fpropOp_name)
                if fpropOp.op_def.name in KFAC_OPS:
                    # Known OPs
                    ###
                    bTensor = [
                        i for i in bpropOp.inputs if 'gradientsSampled' in i.name][-1]
                    bTensorShape = fpropOp.outputs[0].get_shape()
                    if bTensor.get_shape()[0].value == None:
                        bTensor.set_shape(bTensorShape)
                    bTensors.append(bTensor)
                    ###
                    if fpropOp.op_def.name == 'BiasAdd':
                        fTensors = []
                    else:
                        fTensors.append(
                            [i for i in fpropOp.inputs if param.op.name not in i.name][0])
                    fpropOp_name = fpropOp.op_def.name
                else:
                    # unknown OPs, block approximation used
                    bInputsList = [i for i in bpropOp.inputs[
                        0].op.inputs if 'gradientsSampled' in i.name if 'Shape' not in i.name]
                    if len(bInputsList) > 0:
                        bTensor = bInputsList[0]
                        bTensorShape = fpropOp.outputs[0].get_shape()
                        if len(bTensor.get_shape()) > 0 and bTensor.get_shape()[0].value == None:
                            bTensor.set_shape(bTensorShape)
                        bTensors.append(bTensor)
                    fpropOp_name = opTypes.append('UNK-' + fpropOp.op_def.name)

            return {'opName': fpropOp_name, 'op': fpropOp, 'fpropFactors': fTensors, 'bpropFactors': bTensors}

        for t, param in zip(g, varlist):
            if KFAC_DEBUG:
                print(('get factor for '+param.name))
            factors = searchFactors(t, graph)
            factorTensors[param] = factors

        ########
        # check associated weights and bias for homogeneous coordinate representation
        # and check redundent factors
        # TO-DO: there may be a bug to detect associate bias and weights for
        # forking layer, e.g. in inception models.
        for param in varlist:
            factorTensors[param]['assnWeights'] = None
            factorTensors[param]['assnBias'] = None
        for param in varlist:
            if factorTensors[param]['opName'] == 'BiasAdd':
                factorTensors[param]['assnWeights'] = None
                for item in varlist:
                    if len(factorTensors[item]['bpropFactors']) > 0:
                        if (set(factorTensors[item]['bpropFactors']) == set(factorTensors[param]['bpropFactors'])) and (len(factorTensors[item]['fpropFactors']) > 0):
                            factorTensors[param]['assnWeights'] = item
                            factorTensors[item]['assnBias'] = param
                            factorTensors[param]['bpropFactors'] = factorTensors[
                                item]['bpropFactors']

        ########

        ########
        # concatenate the additive gradients along the batch dimension, i.e.
        # assuming independence structure
        for key in ['fpropFactors', 'bpropFactors']:
            for i, param in enumerate(varlist):
                if len(factorTensors[param][key]) > 0:
                    if (key + '_concat') not in factorTensors[param]:
                        name_scope = factorTensors[param][key][0].name.split(':')[
                            0]
                        with tf.name_scope(name_scope):
                            factorTensors[param][
                                key + '_concat'] = tf.concat(factorTensors[param][key], 0)
                else:
                    factorTensors[param][key + '_concat'] = None
                for j, param2 in enumerate(varlist[(i + 1):]):
                    if (len(factorTensors[param][key]) > 0) and (set(factorTensors[param2][key]) == set(factorTensors[param][key])):
                        factorTensors[param2][key] = factorTensors[param][key]
                        factorTensors[param2][
                            key + '_concat'] = factorTensors[param][key + '_concat']
        ########

        if KFAC_DEBUG:
            for items in zip(varlist, fpropTensors, bpropTensors, opTypes):
                print((items[0].name, factorTensors[item]))
        self.factors = factorTensors
        return factorTensors

    def getStats(self, factors, varlist):
        if len(self.stats) == 0:
            # initialize stats variables on CPU because eigen decomp is
            # computed on CPU
            with tf.device('/cpu'):
                tmpStatsCache = {}

                # search for tensor factors and
                # use block diag approx for the bias units
                for var in varlist:
                    fpropFactor = factors[var]['fpropFactors_concat']
                    bpropFactor = factors[var]['bpropFactors_concat']
                    opType = factors[var]['opName']
                    if opType == 'Conv2D':
                        Kh = var.get_shape()[0]
                        Kw = var.get_shape()[1]
                        C = fpropFactor.get_shape()[-1]

                        Oh = bpropFactor.get_shape()[1]
                        Ow = bpropFactor.get_shape()[2]
                        if Oh == 1 and Ow == 1 and self._channel_fac:
                            # factorization along the channels do not support
                            # homogeneous coordinate
                            var_assnBias = factors[var]['assnBias']
                            if var_assnBias:
                                factors[var]['assnBias'] = None
                                factors[var_assnBias]['assnWeights'] = None
                ##

                for var in varlist:
                    fpropFactor = factors[var]['fpropFactors_concat']
                    bpropFactor = factors[var]['bpropFactors_concat']
                    opType = factors[var]['opName']
                    self.stats[var] = {'opName': opType,
                                       'fprop_concat_stats': [],
                                       'bprop_concat_stats': [],
                                       'assnWeights': factors[var]['assnWeights'],
                                       'assnBias': factors[var]['assnBias'],
                                       }
                    if fpropFactor is not None:
                        if fpropFactor not in tmpStatsCache:
                            if opType == 'Conv2D':
                                Kh = var.get_shape()[0]
                                Kw = var.get_shape()[1]
                                C = fpropFactor.get_shape()[-1]

                                Oh = bpropFactor.get_shape()[1]
                                Ow = bpropFactor.get_shape()[2]
                                if Oh == 1 and Ow == 1 and self._channel_fac:
                                    # factorization along the channels
                                    # assume independence between input channels and spatial
                                    # 2K-1 x 2K-1 covariance matrix and C x C covariance matrix
                                    # factorization along the channels do not
                                    # support homogeneous coordinate, assnBias
                                    # is always None
                                    fpropFactor2_size = Kh * Kw
                                    slot_fpropFactor_stats2 = tf.Variable(tf.diag(tf.ones(
                                        [fpropFactor2_size])) * self._diag_init_coeff, name='KFAC_STATS/' + fpropFactor.op.name, trainable=False)
                                    self.stats[var]['fprop_concat_stats'].append(
                                        slot_fpropFactor_stats2)

                                    fpropFactor_size = C
                                else:
                                    # 2K-1 x 2K-1 x C x C covariance matrix
                                    # assume BHWC
                                    fpropFactor_size = Kh * Kw * C
                            else:
                                # D x D covariance matrix
                                fpropFactor_size = fpropFactor.get_shape()[-1]

                            # use homogeneous coordinate
                            if not self._blockdiag_bias and self.stats[var]['assnBias']:
                                fpropFactor_size += 1

                            slot_fpropFactor_stats = tf.Variable(tf.diag(tf.ones(
                                [fpropFactor_size])) * self._diag_init_coeff, name='KFAC_STATS/' + fpropFactor.op.name, trainable=False)
                            self.stats[var]['fprop_concat_stats'].append(
                                slot_fpropFactor_stats)
                            if opType != 'Conv2D':
                                tmpStatsCache[fpropFactor] = self.stats[
                                    var]['fprop_concat_stats']
                        else:
                            self.stats[var][
                                'fprop_concat_stats'] = tmpStatsCache[fpropFactor]

                    if bpropFactor is not None:
                        # no need to collect backward stats for bias vectors if
                        # using homogeneous coordinates
                        if not((not self._blockdiag_bias) and self.stats[var]['assnWeights']):
                            if bpropFactor not in tmpStatsCache:
                                slot_bpropFactor_stats = tf.Variable(tf.diag(tf.ones([bpropFactor.get_shape(
                                )[-1]])) * self._diag_init_coeff, name='KFAC_STATS/' + bpropFactor.op.name, trainable=False)
                                self.stats[var]['bprop_concat_stats'].append(
                                    slot_bpropFactor_stats)
                                tmpStatsCache[bpropFactor] = self.stats[
                                    var]['bprop_concat_stats']
                            else:
                                self.stats[var][
                                    'bprop_concat_stats'] = tmpStatsCache[bpropFactor]

        return self.stats

    def compute_and_apply_stats(self, loss_sampled, var_list=None):
        varlist = var_list
        if varlist is None:
            varlist = tf.trainable_variables()

        stats = self.compute_stats(loss_sampled, var_list=varlist)
        return self.apply_stats(stats)

    def compute_stats(self, loss_sampled, var_list=None):
        varlist = var_list
        if varlist is None:
            varlist = tf.trainable_variables()

        gs = tf.gradients(loss_sampled, varlist, name='gradientsSampled')
        self.gs = gs
        factors = self.getFactors(gs, varlist)
        stats = self.getStats(factors, varlist)

        updateOps = []
        statsUpdates = {}
        statsUpdates_cache = {}
        for var in varlist:
            opType = factors[var]['opName']
            fops = factors[var]['op']
            fpropFactor = factors[var]['fpropFactors_concat']
            fpropStats_vars = stats[var]['fprop_concat_stats']
            bpropFactor = factors[var]['bpropFactors_concat']
            bpropStats_vars = stats[var]['bprop_concat_stats']
            SVD_factors = {}
            for stats_var in fpropStats_vars:
                stats_var_dim = int(stats_var.get_shape()[0])
                if stats_var not in statsUpdates_cache:
                    old_fpropFactor = fpropFactor
                    B = (tf.shape(fpropFactor)[0])  # batch size
                    if opType == 'Conv2D':
                        strides = fops.get_attr("strides")
                        padding = fops.get_attr("padding")
                        convkernel_size = var.get_shape()[0:3]

                        KH = int(convkernel_size[0])
                        KW = int(convkernel_size[1])
                        C = int(convkernel_size[2])
                        flatten_size = int(KH * KW * C)

                        Oh = int(bpropFactor.get_shape()[1])
                        Ow = int(bpropFactor.get_shape()[2])

                        if Oh == 1 and Ow == 1 and self._channel_fac:
                                # factorization along the channels
                                # assume independence among input channels
                                # factor = B x 1 x 1 x (KH xKW x C)
                                # patches = B x Oh x Ow x (KH xKW x C)
                            if len(SVD_factors) == 0:
                                if KFAC_DEBUG:
                                    print(('approx %s act factor with rank-1 SVD factors' % (var.name)))
                                # find closest rank-1 approx to the feature map
                                S, U, V = tf.batch_svd(tf.reshape(
                                    fpropFactor, [-1, KH * KW, C]))
                                # get rank-1 approx slides
                                sqrtS1 = tf.expand_dims(tf.sqrt(S[:, 0, 0]), 1)
                                patches_k = U[:, :, 0] * sqrtS1  # B x KH*KW
                                full_factor_shape = fpropFactor.get_shape()
                                patches_k.set_shape(
                                    [full_factor_shape[0], KH * KW])
                                patches_c = V[:, :, 0] * sqrtS1  # B x C
                                patches_c.set_shape([full_factor_shape[0], C])
                                SVD_factors[C] = patches_c
                                SVD_factors[KH * KW] = patches_k
                            fpropFactor = SVD_factors[stats_var_dim]

                        else:
                            # poor mem usage implementation
                            patches = tf.extract_image_patches(fpropFactor, ksizes=[1, convkernel_size[
                                                               0], convkernel_size[1], 1], strides=strides, rates=[1, 1, 1, 1], padding=padding)

                            if self._approxT2:
                                if KFAC_DEBUG:
                                    print(('approxT2 act fisher for %s' % (var.name)))
                                # T^2 terms * 1/T^2, size: B x C
                                fpropFactor = tf.reduce_mean(patches, [1, 2])
                            else:
                                # size: (B x Oh x Ow) x C
                                fpropFactor = tf.reshape(
                                    patches, [-1, flatten_size]) / Oh / Ow
                    fpropFactor_size = int(fpropFactor.get_shape()[-1])
                    if stats_var_dim == (fpropFactor_size + 1) and not self._blockdiag_bias:
                        if opType == 'Conv2D' and not self._approxT2:
                            # correct padding for numerical stability (we
                            # divided out OhxOw from activations for T1 approx)
                            fpropFactor = tf.concat([fpropFactor, tf.ones(
                                [tf.shape(fpropFactor)[0], 1]) / Oh / Ow], 1)
                        else:
                            # use homogeneous coordinates
                            fpropFactor = tf.concat(
                                [fpropFactor, tf.ones([tf.shape(fpropFactor)[0], 1])], 1)

                    # average over the number of data points in a batch
                    # divided by B
                    cov = tf.matmul(fpropFactor, fpropFactor,
                                    transpose_a=True) / tf.cast(B, tf.float32)
                    updateOps.append(cov)
                    statsUpdates[stats_var] = cov
                    if opType != 'Conv2D':
                        # HACK: for convolution we recompute fprop stats for
                        # every layer including forking layers
                        statsUpdates_cache[stats_var] = cov

            for stats_var in bpropStats_vars:
                stats_var_dim = int(stats_var.get_shape()[0])
                if stats_var not in statsUpdates_cache:
                    old_bpropFactor = bpropFactor
                    bpropFactor_shape = bpropFactor.get_shape()
                    B = tf.shape(bpropFactor)[0]  # batch size
                    C = int(bpropFactor_shape[-1])  # num channels
                    if opType == 'Conv2D' or len(bpropFactor_shape) == 4:
                        if fpropFactor is not None:
                            if self._approxT2:
                                if KFAC_DEBUG:
                                    print(('approxT2 grad fisher for %s' % (var.name)))
                                bpropFactor = tf.reduce_sum(
                                    bpropFactor, [1, 2])  # T^2 terms * 1/T^2
                            else:
                                bpropFactor = tf.reshape(
                                    bpropFactor, [-1, C]) * Oh * Ow  # T * 1/T terms
                        else:
                            # just doing block diag approx. spatial independent
                            # structure does not apply here. summing over
                            # spatial locations
                            if KFAC_DEBUG:
                                print(('block diag approx fisher for %s' % (var.name)))
                            bpropFactor = tf.reduce_sum(bpropFactor, [1, 2])

                    # assume sampled loss is averaged. TO-DO:figure out better
                    # way to handle this
                    bpropFactor *= tf.to_float(B)
                    ##

                    cov_b = tf.matmul(
                        bpropFactor, bpropFactor, transpose_a=True) / tf.to_float(tf.shape(bpropFactor)[0])

                    updateOps.append(cov_b)
                    statsUpdates[stats_var] = cov_b
                    statsUpdates_cache[stats_var] = cov_b

        if KFAC_DEBUG:
            aKey = list(statsUpdates.keys())[0]
            statsUpdates[aKey] = tf.Print(statsUpdates[aKey],
                                          [tf.convert_to_tensor('step:'),
                                           self.global_step,
                                           tf.convert_to_tensor(
                                               'computing stats'),
                                           ])
        self.statsUpdates = statsUpdates
        return statsUpdates

    def apply_stats(self, statsUpdates):
        """ compute stats and update/apply the new stats to the running average
        """

        def updateAccumStats():
            if self._full_stats_init:
                return tf.cond(tf.greater(self.sgd_step, self._cold_iter), lambda: tf.group(*self._apply_stats(statsUpdates, accumulate=True, accumulateCoeff=1. / self._stats_accum_iter)), tf.no_op)
            else:
                return tf.group(*self._apply_stats(statsUpdates, accumulate=True, accumulateCoeff=1. / self._stats_accum_iter))

        def updateRunningAvgStats(statsUpdates, fac_iter=1):
            # return tf.cond(tf.greater_equal(self.factor_step,
            # tf.convert_to_tensor(fac_iter)), lambda:
            # tf.group(*self._apply_stats(stats_list, varlist)), tf.no_op)
            return tf.group(*self._apply_stats(statsUpdates))

        if self._async_stats:
            # asynchronous stats update
            update_stats = self._apply_stats(statsUpdates)

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
            update_stats_op = tf.cond(tf.greater_equal(
                self.stats_step, self._stats_accum_iter), lambda: updateRunningAvgStats(statsUpdates), updateAccumStats)
        self._update_stats_op = update_stats_op
        return update_stats_op

    def _apply_stats(self, statsUpdates, accumulate=False, accumulateCoeff=0.):
        updateOps = []
        # obtain the stats var list
        for stats_var in statsUpdates:
            stats_new = statsUpdates[stats_var]
            if accumulate:
                # simple superbatch averaging
                update_op = tf.assign_add(
                    stats_var, accumulateCoeff * stats_new, use_locking=True)
            else:
                # exponential running averaging
                update_op = tf.assign(
                    stats_var, stats_var * self._stats_decay, use_locking=True)
                update_op = tf.assign_add(
                    update_op, (1. - self._stats_decay) * stats_new, use_locking=True)
            updateOps.append(update_op)

        with tf.control_dependencies(updateOps):
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
                                       tf.convert_to_tensor(accumulateCoeff),
                                       tf.convert_to_tensor('stat step:'),
                                       self.stats_step, updateOps[0], updateOps[1]]))
        return [stats_step_op, ]

    def getStatsEigen(self, stats=None):
        if len(self.stats_eigen) == 0:
            stats_eigen = {}
            if stats is None:
                stats = self.stats

            tmpEigenCache = {}
            with tf.device('/cpu:0'):
                for var in stats:
                    for key in ['fprop_concat_stats', 'bprop_concat_stats']:
                        for stats_var in stats[var][key]:
                            if stats_var not in tmpEigenCache:
                                stats_dim = stats_var.get_shape()[1].value
                                e = tf.Variable(tf.ones(
                                    [stats_dim]), name='KFAC_FAC/' + stats_var.name.split(':')[0] + '/e', trainable=False)
                                Q = tf.Variable(tf.diag(tf.ones(
                                    [stats_dim])), name='KFAC_FAC/' + stats_var.name.split(':')[0] + '/Q', trainable=False)
                                stats_eigen[stats_var] = {'e': e, 'Q': Q}
                                tmpEigenCache[
                                    stats_var] = stats_eigen[stats_var]
                            else:
                                stats_eigen[stats_var] = tmpEigenCache[
                                    stats_var]
            self.stats_eigen = stats_eigen
        return self.stats_eigen

    def computeStatsEigen(self):
        """ compute the eigen decomp using copied var stats to avoid concurrent read/write from other queue """
        # TO-DO: figure out why this op has delays (possibly moving
        # eigenvectors around?)
        with tf.device('/cpu:0'):
            def removeNone(tensor_list):
                local_list = []
                for item in tensor_list:
                    if item is not None:
                        local_list.append(item)
                return local_list

            def copyStats(var_list):
                print("copying stats to buffer tensors before eigen decomp")
                redundant_stats = {}
                copied_list = []
                for item in var_list:
                    if item is not None:
                        if item not in redundant_stats:
                            if self._use_float64:
                                redundant_stats[item] = tf.cast(
                                    tf.identity(item), tf.float64)
                            else:
                                redundant_stats[item] = tf.identity(item)
                        copied_list.append(redundant_stats[item])
                    else:
                        copied_list.append(None)
                return copied_list
            #stats = [copyStats(self.fStats), copyStats(self.bStats)]
            #stats = [self.fStats, self.bStats]

            stats_eigen = self.stats_eigen
            computedEigen = {}
            eigen_reverse_lookup = {}
            updateOps = []
            # sync copied stats
            # with tf.control_dependencies(removeNone(stats[0]) +
            # removeNone(stats[1])):
            with tf.control_dependencies([]):
                for stats_var in stats_eigen:
                    if stats_var not in computedEigen:
                        eigens = tf.self_adjoint_eig(stats_var)
                        e = eigens[0]
                        Q = eigens[1]
                        if self._use_float64:
                            e = tf.cast(e, tf.float32)
                            Q = tf.cast(Q, tf.float32)
                        updateOps.append(e)
                        updateOps.append(Q)
                        computedEigen[stats_var] = {'e': e, 'Q': Q}
                        eigen_reverse_lookup[e] = stats_eigen[stats_var]['e']
                        eigen_reverse_lookup[Q] = stats_eigen[stats_var]['Q']

            self.eigen_reverse_lookup = eigen_reverse_lookup
            self.eigen_update_list = updateOps

            if KFAC_DEBUG:
                self.eigen_update_list = [item for item in updateOps]
                with tf.control_dependencies(updateOps):
                    updateOps.append(tf.Print(tf.constant(
                        0.), [tf.convert_to_tensor('computed factor eigen')]))

        return updateOps

    def applyStatsEigen(self, eigen_list):
        updateOps = []
        print(('updating %d eigenvalue/vectors' % len(eigen_list)))
        for i, (tensor, mark) in enumerate(zip(eigen_list, self.eigen_update_list)):
            stats_eigen_var = self.eigen_reverse_lookup[mark]
            updateOps.append(
                tf.assign(stats_eigen_var, tensor, use_locking=True))

        with tf.control_dependencies(updateOps):
            factor_step_op = tf.assign_add(self.factor_step, 1)
            updateOps.append(factor_step_op)
            if KFAC_DEBUG:
                updateOps.append(tf.Print(tf.constant(
                    0.), [tf.convert_to_tensor('updated kfac factors')]))
        return updateOps

    def getKfacPrecondUpdates(self, gradlist, varlist):
        updatelist = []
        vg = 0.

        assert len(self.stats) > 0
        assert len(self.stats_eigen) > 0
        assert len(self.factors) > 0
        counter = 0

        grad_dict = {var: grad for grad, var in zip(gradlist, varlist)}

        for grad, var in zip(gradlist, varlist):
            GRAD_RESHAPE = False
            GRAD_TRANSPOSE = False

            fpropFactoredFishers = self.stats[var]['fprop_concat_stats']
            bpropFactoredFishers = self.stats[var]['bprop_concat_stats']

            if (len(fpropFactoredFishers) + len(bpropFactoredFishers)) > 0:
                counter += 1
                GRAD_SHAPE = grad.get_shape()
                if len(grad.get_shape()) > 2:
                    # reshape conv kernel parameters
                    KW = int(grad.get_shape()[0])
                    KH = int(grad.get_shape()[1])
                    C = int(grad.get_shape()[2])
                    D = int(grad.get_shape()[3])

                    if len(fpropFactoredFishers) > 1 and self._channel_fac:
                        # reshape conv kernel parameters into tensor
                        grad = tf.reshape(grad, [KW * KH, C, D])
                    else:
                        # reshape conv kernel parameters into 2D grad
                        grad = tf.reshape(grad, [-1, D])
                    GRAD_RESHAPE = True
                elif len(grad.get_shape()) == 1:
                    # reshape bias or 1D parameters
                    D = int(grad.get_shape()[0])

                    grad = tf.expand_dims(grad, 0)
                    GRAD_RESHAPE = True
                else:
                    # 2D parameters
                    C = int(grad.get_shape()[0])
                    D = int(grad.get_shape()[1])

                if (self.stats[var]['assnBias'] is not None) and not self._blockdiag_bias:
                    # use homogeneous coordinates only works for 2D grad.
                    # TO-DO: figure out how to factorize bias grad
                    # stack bias grad
                    var_assnBias = self.stats[var]['assnBias']
                    grad = tf.concat(
                        [grad, tf.expand_dims(grad_dict[var_assnBias], 0)], 0)

                # project gradient to eigen space and reshape the eigenvalues
                # for broadcasting
                eigVals = []

                for idx, stats in enumerate(self.stats[var]['fprop_concat_stats']):
                    Q = self.stats_eigen[stats]['Q']
                    e = detectMinVal(self.stats_eigen[stats][
                                     'e'], var, name='act', debug=KFAC_DEBUG)

                    Q, e = factorReshape(Q, e, grad, facIndx=idx, ftype='act')
                    eigVals.append(e)
                    grad = gmatmul(Q, grad, transpose_a=True, reduce_dim=idx)

                for idx, stats in enumerate(self.stats[var]['bprop_concat_stats']):
                    Q = self.stats_eigen[stats]['Q']
                    e = detectMinVal(self.stats_eigen[stats][
                                     'e'], var, name='grad', debug=KFAC_DEBUG)

                    Q, e = factorReshape(Q, e, grad, facIndx=idx, ftype='grad')
                    eigVals.append(e)
                    grad = gmatmul(grad, Q, transpose_b=False, reduce_dim=idx)
                ##

                #####
                # whiten using eigenvalues
                weightDecayCoeff = 0.
                if var in self._weight_decay_dict:
                    weightDecayCoeff = self._weight_decay_dict[var]
                    if KFAC_DEBUG:
                        print(('weight decay coeff for %s is %f' % (var.name, weightDecayCoeff)))

                if self._factored_damping:
                    if KFAC_DEBUG:
                        print(('use factored damping for %s' % (var.name)))
                    coeffs = 1.
                    num_factors = len(eigVals)
                    # compute the ratio of two trace norm of the left and right
                    # KFac matrices, and their generalization
                    if len(eigVals) == 1:
                        damping = self._epsilon + weightDecayCoeff
                    else:
                        damping = tf.pow(
                            self._epsilon + weightDecayCoeff, 1. / num_factors)
                    eigVals_tnorm_avg = [tf.reduce_mean(
                        tf.abs(e)) for e in eigVals]
                    for e, e_tnorm in zip(eigVals, eigVals_tnorm_avg):
                        eig_tnorm_negList = [
                            item for item in eigVals_tnorm_avg if item != e_tnorm]
                        if len(eigVals) == 1:
                            adjustment = 1.
                        elif len(eigVals) == 2:
                            adjustment = tf.sqrt(
                                e_tnorm / eig_tnorm_negList[0])
                        else:
                            eig_tnorm_negList_prod = reduce(
                                lambda x, y: x * y, eig_tnorm_negList)
                            adjustment = tf.pow(
                                tf.pow(e_tnorm, num_factors - 1.) / eig_tnorm_negList_prod, 1. / num_factors)
                        coeffs *= (e + adjustment * damping)
                else:
                    coeffs = 1.
                    damping = (self._epsilon + weightDecayCoeff)
                    for e in eigVals:
                        coeffs *= e
                    coeffs += damping

                #grad = tf.Print(grad, [tf.convert_to_tensor('1'), tf.convert_to_tensor(var.name), grad.get_shape()])

                grad /= coeffs

                #grad = tf.Print(grad, [tf.convert_to_tensor('2'), tf.convert_to_tensor(var.name), grad.get_shape()])
                #####
                # project gradient back to euclidean space
                for idx, stats in enumerate(self.stats[var]['fprop_concat_stats']):
                    Q = self.stats_eigen[stats]['Q']
                    grad = gmatmul(Q, grad, transpose_a=False, reduce_dim=idx)

                for idx, stats in enumerate(self.stats[var]['bprop_concat_stats']):
                    Q = self.stats_eigen[stats]['Q']
                    grad = gmatmul(grad, Q, transpose_b=True, reduce_dim=idx)
                ##

                #grad = tf.Print(grad, [tf.convert_to_tensor('3'), tf.convert_to_tensor(var.name), grad.get_shape()])
                if (self.stats[var]['assnBias'] is not None) and not self._blockdiag_bias:
                    # use homogeneous coordinates only works for 2D grad.
                    # TO-DO: figure out how to factorize bias grad
                    # un-stack bias grad
                    var_assnBias = self.stats[var]['assnBias']
                    C_plus_one = int(grad.get_shape()[0])
                    grad_assnBias = tf.reshape(tf.slice(grad,
                                                        begin=[
                                                            C_plus_one - 1, 0],
                                                        size=[1, -1]), var_assnBias.get_shape())
                    grad_assnWeights = tf.slice(grad,
                                                begin=[0, 0],
                                                size=[C_plus_one - 1, -1])
                    grad_dict[var_assnBias] = grad_assnBias
                    grad = grad_assnWeights

                #grad = tf.Print(grad, [tf.convert_to_tensor('4'), tf.convert_to_tensor(var.name), grad.get_shape()])
                if GRAD_RESHAPE:
                    grad = tf.reshape(grad, GRAD_SHAPE)

                grad_dict[var] = grad

        print(('projecting %d gradient matrices' % counter))

        for g, var in zip(gradlist, varlist):
            grad = grad_dict[var]
            ### clipping ###
            if KFAC_DEBUG:
                print(('apply clipping to %s' % (var.name)))
            tf.Print(grad, [tf.sqrt(tf.reduce_sum(tf.pow(grad, 2)))], "Euclidean norm of new grad")
            local_vg = tf.reduce_sum(grad * g * (self._lr * self._lr))
            vg += local_vg

        # recale everything
        if KFAC_DEBUG:
            print('apply vFv clipping')

        scaling = tf.minimum(1., tf.sqrt(self._clip_kl / vg))
        if KFAC_DEBUG:
            scaling = tf.Print(scaling, [tf.convert_to_tensor(
                'clip: '), scaling, tf.convert_to_tensor(' vFv: '), vg])
        with tf.control_dependencies([tf.assign(self.vFv, vg)]):
            updatelist = [grad_dict[var] for var in varlist]
            for i, item in enumerate(updatelist):
                updatelist[i] = scaling * item

        return updatelist

    def compute_gradients(self, loss, var_list=None):
        varlist = var_list
        if varlist is None:
            varlist = tf.trainable_variables()
        g = tf.gradients(loss, varlist)

        return [(a, b) for a, b in zip(g, varlist)]

    def apply_gradients_kfac(self, grads):
        g, varlist = list(zip(*grads))

        if len(self.stats_eigen) == 0:
            self.getStatsEigen()

        qr = None
        # launch eigen-decomp on a queue thread
        if self._async:
            print('Use async eigen decomp')
            # get a list of factor loading tensors
            factorOps_dummy = self.computeStatsEigen()

            # define a queue for the list of factor loading tensors
            queue = tf.FIFOQueue(1, [item.dtype for item in factorOps_dummy], shapes=[
                                 item.get_shape() for item in factorOps_dummy])
            enqueue_op = tf.cond(tf.logical_and(tf.equal(tf.mod(self.stats_step, self._kfac_update), tf.convert_to_tensor(
                0)), tf.greater_equal(self.stats_step, self._stats_accum_iter)), lambda: queue.enqueue(self.computeStatsEigen()), tf.no_op)

            def dequeue_op():
                return queue.dequeue()

            qr = tf.train.QueueRunner(queue, [enqueue_op])

        updateOps = []
        global_step_op = tf.assign_add(self.global_step, 1)
        updateOps.append(global_step_op)

        with tf.control_dependencies([global_step_op]):

            # compute updates
            assert self._update_stats_op != None
            updateOps.append(self._update_stats_op)
            dependency_list = []
            if not self._async:
                dependency_list.append(self._update_stats_op)

            with tf.control_dependencies(dependency_list):
                def no_op_wrapper():
                    return tf.group(*[tf.assign_add(self.cold_step, 1)])

                if not self._async:
                    # synchronous eigen-decomp updates
                    updateFactorOps = tf.cond(tf.logical_and(tf.equal(tf.mod(self.stats_step, self._kfac_update),
                                                                      tf.convert_to_tensor(0)),
                                                             tf.greater_equal(self.stats_step, self._stats_accum_iter)), lambda: tf.group(*self.applyStatsEigen(self.computeStatsEigen())), no_op_wrapper)
                else:
                    # asynchronous eigen-decomp updates using queue
                    updateFactorOps = tf.cond(tf.greater_equal(self.stats_step, self._stats_accum_iter),
                                              lambda: tf.cond(tf.equal(queue.size(), tf.convert_to_tensor(0)),
                                                              tf.no_op,

                                                              lambda: tf.group(
                                                                  *self.applyStatsEigen(dequeue_op())),
                                                              ),
                                              no_op_wrapper)

                updateOps.append(updateFactorOps)

                with tf.control_dependencies([updateFactorOps]):
                    def gradOp():
                        return list(g)

                    def getKfacGradOp():
                        return self.getKfacPrecondUpdates(g, varlist)
                    u = tf.cond(tf.greater(self.factor_step,
                                           tf.convert_to_tensor(0)), getKfacGradOp, gradOp)

                    optim = tf.train.MomentumOptimizer(
                        self._lr * (1. - self._momentum), self._momentum)
                    #optim = tf.train.AdamOptimizer(self._lr, epsilon=0.01)

                    def optimOp():
                        def updateOptimOp():
                            if self._full_stats_init:
                                return tf.cond(tf.greater(self.factor_step, tf.convert_to_tensor(0)), lambda: optim.apply_gradients(list(zip(u, varlist))), tf.no_op)
                            else:
                                return optim.apply_gradients(list(zip(u, varlist)))
                        if self._full_stats_init:
                            return tf.cond(tf.greater_equal(self.stats_step, self._stats_accum_iter), updateOptimOp, tf.no_op)
                        else:
                            return tf.cond(tf.greater_equal(self.sgd_step, self._cold_iter), updateOptimOp, tf.no_op)
                    updateOps.append(optimOp())

        return tf.group(*updateOps), qr

    def apply_gradients(self, grads):
        coldOptim = tf.train.MomentumOptimizer(
            self._cold_lr, self._momentum)

        def coldSGDstart():
            sgd_grads, sgd_var = zip(*grads)

            if self.max_grad_norm != None:
                sgd_grads, sgd_grad_norm = tf.clip_by_global_norm(sgd_grads,self.max_grad_norm)

            sgd_grads = list(zip(sgd_grads,sgd_var))

            sgd_step_op = tf.assign_add(self.sgd_step, 1)
            coldOptim_op = coldOptim.apply_gradients(sgd_grads)
            if KFAC_DEBUG:
                with tf.control_dependencies([sgd_step_op, coldOptim_op]):
                    sgd_step_op = tf.Print(
                        sgd_step_op, [self.sgd_step, tf.convert_to_tensor('doing cold sgd step')])
            return tf.group(*[sgd_step_op, coldOptim_op])

        kfacOptim_op, qr = self.apply_gradients_kfac(grads)

        def warmKFACstart():
            return kfacOptim_op

        return tf.cond(tf.greater(self.sgd_step, self._cold_iter), warmKFACstart, coldSGDstart), qr

    def minimize(self, loss, loss_sampled, var_list=None):
        grads = self.compute_gradients(loss, var_list=var_list)
        update_stats_op = self.compute_and_apply_stats(
            loss_sampled, var_list=var_list)
        return self.apply_gradients(grads)
