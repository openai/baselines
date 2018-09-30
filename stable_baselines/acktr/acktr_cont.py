"""
Continuous acktr
"""

import numpy as np
import tensorflow as tf

from stable_baselines import logger
import stable_baselines.common as common
from stable_baselines.common import tf_util
from stable_baselines.acktr import kfac
from stable_baselines.common.filters import ZFilter


def rollout(env, policy, max_pathlength, animate=False, obfilter=None):
    """
    Simulate the env and policy for max_pathlength steps

    :param env: (Gym environment) The environment to learn from
    :param policy: (Object) The policy model to use (MLP, CNN, LSTM, ...)
    :param max_pathlength: (int) The maximum length for an episode
    :param animate: (bool) if render env
    :param obfilter: (Filter) the observation filter
    :return: (dict) observation, terminated, reward, action, action_dist, logp
    """
    observation = env.reset()
    prev_ob = np.float32(np.zeros(observation.shape))
    if obfilter:
        observation = obfilter(observation)
    terminated = False

    observations = []
    actions = []
    action_dists = []
    logps = []
    rewards = []
    for _ in range(max_pathlength):
        if animate:
            env.render()
        state = np.concatenate([observation, prev_ob], -1)
        observations.append(state)
        action, ac_dist, logp = policy.act(state)
        actions.append(action)
        action_dists.append(ac_dist)
        logps.append(logp)
        prev_ob = np.copy(observation)
        scaled_ac = env.action_space.low + (action + 1.) * 0.5 * (env.action_space.high - env.action_space.low)
        scaled_ac = np.clip(scaled_ac, env.action_space.low, env.action_space.high)
        observation, rew, done, _ = env.step(scaled_ac)
        if obfilter:
            observation = obfilter(observation)
        rewards.append(rew)
        if done:
            terminated = True
            break
    return {"observation": np.array(observations), "terminated": terminated,
            "reward": np.array(rewards), "action": np.array(actions),
            "action_dist": np.array(action_dists), "logp": np.array(logps)}


def learn(env, policy, value_fn, gamma, lam, timesteps_per_batch, num_timesteps,
          animate=False, callback=None, desired_kl=0.002):
    """
    Trains an ACKTR model.

    :param env: (Gym environment) The environment to learn from
    :param policy: (Object) The policy model to use (MLP, CNN, LSTM, ...)
    :param value_fn: (Object) The value function model to use (MLP, CNN, LSTM, ...)
    :param gamma: (float) The discount value
    :param lam: (float) the tradeoff between exploration and exploitation
    :param timesteps_per_batch: (int) the number of timesteps for each batch
    :param num_timesteps: (int) the total number of timesteps to run
    :param animate: (bool) if render env
    :param callback: (function) called every step, used for logging and saving
    :param desired_kl: (float) the Kullback leibler weight for the loss
    """
    obfilter = ZFilter(env.observation_space.shape)

    max_pathlength = env.spec.timestep_limit
    stepsize = tf.Variable(initial_value=np.float32(np.array(0.03)), name='stepsize')
    inputs, loss, loss_sampled = policy.update_info
    optim = kfac.KfacOptimizer(learning_rate=stepsize, cold_lr=stepsize * (1 - 0.9), momentum=0.9, kfac_update=2,
                               epsilon=1e-2, stats_decay=0.99, async_eigen_decomp=1, cold_iter=1,
                               weight_decay_dict=policy.wd_dict, max_grad_norm=None)
    pi_var_list = []
    for var in tf.trainable_variables():
        if "pi" in var.name:
            pi_var_list.append(var)

    update_op, q_runner = optim.minimize(loss, loss_sampled, var_list=pi_var_list)
    do_update = tf_util.function(inputs, update_op)
    tf_util.initialize()

    # start queue runners
    enqueue_threads = []
    coord = tf.train.Coordinator()
    for queue_runner in [q_runner, value_fn.q_runner]:
        assert queue_runner is not None
        enqueue_threads.extend(queue_runner.create_threads(tf.get_default_session(), coord=coord, start=True))

    i = 0
    timesteps_so_far = 0
    while True:
        if timesteps_so_far > num_timesteps:
            break
        logger.log("********** Iteration %i ************" % i)

        # Collect paths until we have enough timesteps
        timesteps_this_batch = 0
        paths = []
        while True:
            path = rollout(env, policy, max_pathlength, animate=(len(paths) == 0 and (i % 10 == 0) and animate),
                           obfilter=obfilter)
            paths.append(path)
            timesteps_this_batch += path["reward"].shape[0]
            timesteps_so_far += path["reward"].shape[0]
            if timesteps_this_batch > timesteps_per_batch:
                break

        # Estimate advantage function
        vtargs = []
        advs = []
        for path in paths:
            rew_t = path["reward"]
            return_t = common.discount(rew_t, gamma)
            vtargs.append(return_t)
            vpred_t = value_fn.predict(path)
            vpred_t = np.append(vpred_t, 0.0 if path["terminated"] else vpred_t[-1])
            delta_t = rew_t + gamma * vpred_t[1:] - vpred_t[:-1]
            adv_t = common.discount(delta_t, gamma * lam)
            advs.append(adv_t)
        # Update value function
        value_fn.fit(paths, vtargs)

        # Build arrays for policy update
        ob_no = np.concatenate([path["observation"] for path in paths])
        action_na = np.concatenate([path["action"] for path in paths])
        oldac_dist = np.concatenate([path["action_dist"] for path in paths])
        adv_n = np.concatenate(advs)
        standardized_adv_n = (adv_n - adv_n.mean()) / (adv_n.std() + 1e-8)

        # Policy update
        do_update(ob_no, action_na, standardized_adv_n)

        min_stepsize = np.float32(1e-8)
        max_stepsize = np.float32(1e0)
        # Adjust stepsize
        kl_loss = policy.compute_kl(ob_no, oldac_dist)
        if kl_loss > desired_kl * 2:
            logger.log("kl too high")
            tf.assign(stepsize, tf.maximum(min_stepsize, stepsize / 1.5)).eval()
        elif kl_loss < desired_kl / 2:
            logger.log("kl too low")
            tf.assign(stepsize, tf.minimum(max_stepsize, stepsize * 1.5)).eval()
        else:
            logger.log("kl just right!")

        logger.record_tabular("EpRewMean", np.mean([path["reward"].sum() for path in paths]))
        logger.record_tabular("EpRewSEM", np.std([path["reward"].sum() / np.sqrt(len(paths)) for path in paths]))
        logger.record_tabular("EpLenMean", np.mean([path["reward"].shape[0] for path in paths]))
        logger.record_tabular("KL", kl_loss)
        if callback:
            callback()
        logger.dump_tabular()
        i += 1

    coord.request_stop()
    coord.join(enqueue_threads)
