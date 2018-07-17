from collections import deque
import time

import tensorflow as tf
import numpy as np
from mpi4py import MPI

from baselines.common import Dataset, explained_variance, fmt_row, zipsame
from baselines import logger
import baselines.common.tf_util as tf_util
from baselines.common.mpi_adam import MpiAdam
from baselines.common.mpi_moments import mpi_moments
from baselines.gail.trpo_mpi import traj_segment_generator, add_vtarg_and_adv, flatten_lists


def learn(env, policy_fn, *, timesteps_per_actorbatch, clip_param, entcoeff, optim_epochs, optim_stepsize,
          optim_batchsize, gamma, lam, max_timesteps=0, max_episodes=0, max_iters=0, max_seconds=0, callback=None,
          adam_epsilon=1e-5, schedule='constant'):
    """
    Learning PPO with Stochastic Gradient Descent

    :param env: (Gym Environment) environment to train on
    :param policy_fn: (function (str, Gym Spaces, Gym Spaces): TensorFlow Tensor) creates the policy
    :param timesteps_per_actorbatch: (int) timesteps per actor per update
    :param clip_param: (float) clipping parameter epsilon
    :param entcoeff: (float) the entropy loss weight
    :param optim_epochs: (float) the optimizer's number of epochs
    :param optim_stepsize: (float) the optimizer's stepsize
    :param optim_batchsize: (int) the optimizer's the batch size
    :param gamma: (float) discount factor
    :param lam: (float) advantage estimation
    :param max_timesteps: (int) number of env steps to optimizer for
    :param max_episodes: (int) the maximum number of epochs
    :param max_iters: (int) the maximum number of iterations
    :param max_seconds: (int) the maximal duration
    :param callback: (function (dict, dict)) function called at every steps with state of the algorithm.
        It takes the local and global variables.
    :param adam_epsilon: (float) the epsilon value for the adam optimizer
    :param schedule: (str) The type of scheduler for the learning rate update ('linear', 'constant',
                                 'double_linear_con', 'middle_drop' or 'double_middle_drop')
    """

    # Setup losses and stuff
    ob_space = env.observation_space
    ac_space = env.action_space
    sess = tf_util.single_threaded_session()

    # Construct network for new policy
    policy = policy_fn("pi", ob_space, ac_space, sess=sess)

    # Network for old policy
    oldpi = policy_fn("oldpi", ob_space, ac_space, sess=sess)

    # Target advantage function (if applicable)
    atarg = tf.placeholder(dtype=tf.float32, shape=[None])

    # Empirical return
    ret = tf.placeholder(dtype=tf.float32, shape=[None])

    # learning rate multiplier, updated with schedule
    lrmult = tf.placeholder(name='lrmult', dtype=tf.float32, shape=[])

    # Annealed cliping parameter epislon
    clip_param = clip_param * lrmult

    obs_ph = tf_util.get_placeholder_cached(name="ob")
    action_ph = policy.pdtype.sample_placeholder([None])

    kloldnew = oldpi.proba_distribution.kl(policy.proba_distribution)
    ent = policy.proba_distribution.entropy()
    meankl = tf.reduce_mean(kloldnew)
    meanent = tf.reduce_mean(ent)
    pol_entpen = (-entcoeff) * meanent

    # pnew / pold
    ratio = tf.exp(policy.proba_distribution.logp(action_ph) - oldpi.proba_distribution.logp(action_ph))

    # surrogate from conservative policy iteration
    surr1 = ratio * atarg
    surr2 = tf.clip_by_value(ratio, 1.0 - clip_param, 1.0 + clip_param) * atarg

    # PPO's pessimistic surrogate (L^CLIP)
    pol_surr = - tf.reduce_mean(tf.minimum(surr1, surr2))
    vf_loss = tf.reduce_mean(tf.square(policy.vpred - ret))
    total_loss = pol_surr + pol_entpen + vf_loss
    losses = [pol_surr, pol_entpen, vf_loss, meankl, meanent]
    loss_names = ["pol_surr", "pol_entpen", "vf_loss", "kl", "ent"]

    var_list = policy.get_trainable_variables()
    lossandgrad = tf_util.function([obs_ph, action_ph, atarg, ret, lrmult],
                                   losses + [tf_util.flatgrad(total_loss, var_list)])
    adam = MpiAdam(var_list, epsilon=adam_epsilon, sess=sess)

    assign_old_eq_new = tf_util.function([], [], updates=[tf.assign(oldv, newv)
                                                          for (oldv, newv) in
                                                          zipsame(oldpi.get_variables(), policy.get_variables())])
    compute_losses = tf_util.function([obs_ph, action_ph, atarg, ret, lrmult], losses)

    tf_util.initialize(sess=sess)
    adam.sync()

    # Prepare for rollouts
    seg_gen = traj_segment_generator(policy, env, timesteps_per_actorbatch, stochastic=True)

    episodes_so_far = 0
    timesteps_so_far = 0
    iters_so_far = 0
    tstart = time.time()

    # rolling buffer for episode lengths
    lenbuffer = deque(maxlen=100)
    # rolling buffer for episode rewards
    rewbuffer = deque(maxlen=100)

    assert sum([max_iters > 0, max_timesteps > 0, max_episodes > 0,
                max_seconds > 0]) == 1, "Only one time constraint permitted"

    while True:
        if callback:
            callback(locals(), globals())
        if max_timesteps and timesteps_so_far >= max_timesteps:
            break
        elif max_episodes and episodes_so_far >= max_episodes:
            break
        elif max_iters and iters_so_far >= max_iters:
            break
        elif max_seconds and time.time() - tstart >= max_seconds:
            break

        if schedule == 'constant':
            cur_lrmult = 1.0
        elif schedule == 'linear':
            cur_lrmult = max(1.0 - float(timesteps_so_far) / max_timesteps, 0)
        else:
            raise NotImplementedError

        logger.log("********** Iteration %i ************" % iters_so_far)

        seg = seg_gen.__next__()
        add_vtarg_and_adv(seg, gamma, lam)

        # ob, ac, atarg, ret, td1ret = map(np.concatenate, (obs, acs, atargs, rets, td1rets))
        obs_ph, action_ph, atarg, tdlamret = seg["ob"], seg["ac"], seg["adv"], seg["tdlamret"]

        # predicted value function before udpate
        vpredbefore = seg["vpred"]

        # standardized advantage function estimate
        atarg = (atarg - atarg.mean()) / atarg.std()
        dataset = Dataset(dict(ob=obs_ph, ac=action_ph, atarg=atarg, vtarg=tdlamret),
                          shuffle=not policy.recurrent)
        optim_batchsize = optim_batchsize or obs_ph.shape[0]

        if hasattr(policy, "ob_rms"):
            # update running mean/std for policy
            policy.ob_rms.update(obs_ph)

        # set old parameter values to new parameter values
        assign_old_eq_new(sess=sess)
        logger.log("Optimizing...")
        logger.log(fmt_row(13, loss_names))

        # Here we do a bunch of optimization epochs over the data
        for _ in range(optim_epochs):
            # list of tuples, each of which gives the loss for a minibatch
            losses = []
            for batch in dataset.iterate_once(optim_batchsize):
                *newlosses, grad = lossandgrad(batch["ob"], batch["ac"], batch["atarg"], batch["vtarg"], cur_lrmult,
                                               sess=sess)
                adam.update(grad, optim_stepsize * cur_lrmult)
                losses.append(newlosses)
            logger.log(fmt_row(13, np.mean(losses, axis=0)))

        logger.log("Evaluating losses...")
        losses = []
        for batch in dataset.iterate_once(optim_batchsize):
            newlosses = compute_losses(batch["ob"], batch["ac"], batch["atarg"], batch["vtarg"], cur_lrmult, sess=sess)
            losses.append(newlosses)
        meanlosses, _, _ = mpi_moments(losses, axis=0)
        logger.log(fmt_row(13, meanlosses))
        for (lossval, name) in zipsame(meanlosses, loss_names):
            logger.record_tabular("loss_" + name, lossval)
        logger.record_tabular("ev_tdlam_before", explained_variance(vpredbefore, tdlamret))

        # local values
        lrlocal = (seg["ep_lens"], seg["ep_rets"])

        # list of tuples
        listoflrpairs = MPI.COMM_WORLD.allgather(lrlocal)
        lens, rews = map(flatten_lists, zip(*listoflrpairs))
        lenbuffer.extend(lens)
        rewbuffer.extend(rews)
        logger.record_tabular("EpLenMean", np.mean(lenbuffer))
        logger.record_tabular("EpRewMean", np.mean(rewbuffer))
        logger.record_tabular("EpThisIter", len(lens))
        episodes_so_far += len(lens)
        timesteps_so_far += sum(lens)
        iters_so_far += 1
        logger.record_tabular("EpisodesSoFar", episodes_so_far)
        logger.record_tabular("TimestepsSoFar", timesteps_so_far)
        logger.record_tabular("TimeElapsed", time.time() - tstart)
        if MPI.COMM_WORLD.Get_rank() == 0:
            logger.dump_tabular()

    return policy
