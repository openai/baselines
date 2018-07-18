"""
Discrete acktr
"""

import os
import time
import joblib

import tensorflow as tf

from baselines import logger
from baselines.common import set_global_seeds, explained_variance
from baselines.a2c.a2c import A2CRunner
from baselines.a2c.utils import Scheduler, find_trainable_variables, calc_entropy, mse
from baselines.acktr import kfac


class Model(object):
    def __init__(self, policy, ob_space, ac_space, n_envs, total_timesteps, nprocs=32, n_steps=20,
                 ent_coef=0.01, vf_coef=0.25, vf_fisher_coef=1.0, learning_rate=0.25, max_grad_norm=0.5,
                 kfac_clip=0.001, lr_schedule='linear'):
        """
        The ACKTR (Actor Critic using Kronecker-Factored Trust Region) model class, https://arxiv.org/abs/1708.05144

        :param policy: (Object) The policy model to use (MLP, CNN, LSTM, ...)
        :param ob_space: (Gym Space) The observation space
        :param ac_space: (Gym Space) The action space
        :param n_envs: (int) The number of environments
        :param total_timesteps: (int) The total number of timesteps for training the model
        :param nprocs: (int) The number of threads for TensorFlow operations
        :param n_steps: (int) The number of steps to run for each environment
        :param ent_coef: (float) The weight for the entropic loss
        :param vf_coef: (float) The weight for the loss on the value function
        :param vf_fisher_coef: (float) The weight for the fisher loss on the value function
        :param learning_rate: (float) The initial learning rate for the RMS prop optimizer
        :param max_grad_norm: (float) The clipping value for the maximum gradient
        :param kfac_clip: (float) gradient clipping for Kullback leiber
        :param lr_schedule: (str) The type of scheduler for the learning rate update ('linear', 'constant',
                                 'double_linear_con', 'middle_drop' or 'double_middle_drop')
        """

        config = tf.ConfigProto(allow_soft_placement=True,
                                intra_op_parallelism_threads=nprocs,
                                inter_op_parallelism_threads=nprocs)
        config.gpu_options.allow_growth = True
        self.sess = sess = tf.Session(config=config)
        n_batch = n_envs * n_steps
        action_ph = tf.placeholder(tf.int32, [n_batch])
        advs_ph = tf.placeholder(tf.float32, [n_batch])
        rewards_ph = tf.placeholder(tf.float32, [n_batch])
        pg_lr_ph = tf.placeholder(tf.float32, [])

        self.model = step_model = policy(sess, ob_space, ac_space, n_envs, 1, reuse=False)
        self.model2 = train_model = policy(sess, ob_space, ac_space, n_envs * n_steps, n_steps, reuse=True)

        logpac = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=train_model.policy, labels=action_ph)
        self.logits = train_model.policy

        # training loss
        pg_loss = tf.reduce_mean(advs_ph * logpac)
        entropy = tf.reduce_mean(calc_entropy(train_model.policy))
        pg_loss = pg_loss - ent_coef * entropy
        vf_loss = mse(tf.squeeze(train_model.value_fn), rewards_ph)
        train_loss = pg_loss + vf_coef * vf_loss

        # Fisher loss construction
        self.pg_fisher = pg_fisher_loss = -tf.reduce_mean(logpac)
        sample_net = train_model.value_fn + tf.random_normal(tf.shape(train_model.value_fn))
        self.vf_fisher = vf_fisher_loss = - vf_fisher_coef * tf.reduce_mean(
            tf.pow(train_model.value_fn - tf.stop_gradient(sample_net), 2))
        self.joint_fisher = pg_fisher_loss + vf_fisher_loss

        self.params = params = find_trainable_variables("model")

        self.grads_check = grads = tf.gradients(train_loss, params)

        with tf.device('/gpu:0'):
            self.optim = optim = kfac.KfacOptimizer(learning_rate=pg_lr_ph, clip_kl=kfac_clip,
                                                    momentum=0.9, kfac_update=1, epsilon=0.01,
                                                    stats_decay=0.99, async=1, cold_iter=10,
                                                    max_grad_norm=max_grad_norm)

            optim.compute_and_apply_stats(self.joint_fisher, var_list=params)
            train_op, q_runner = optim.apply_gradients(list(zip(grads, params)))
        self.q_runner = q_runner
        self.learning_rate = Scheduler(initial_value=learning_rate, n_values=total_timesteps, schedule=lr_schedule)

        def train(obs, states, rewards, masks, actions, values):
            advs = rewards - values
            for _ in range(len(obs)):
                cur_lr = self.learning_rate.value()

            td_map = {train_model.obs_ph: obs, action_ph: actions, advs_ph: advs, rewards_ph: rewards, pg_lr_ph: cur_lr}
            if states is not None:
                td_map[train_model.states_ph] = states
                td_map[train_model.masks_ph] = masks

            policy_loss, value_loss, policy_entropy, _ = sess.run(
                [pg_loss, vf_loss, entropy, train_op],
                td_map
            )
            return policy_loss, value_loss, policy_entropy

        def save(save_path):
            session_params = sess.run(params)
            joblib.dump(session_params, save_path)

        def load(load_path):
            loaded_params = joblib.load(load_path)
            restores = []
            for param, loaded_p in zip(params, loaded_params):
                restores.append(param.assign(loaded_p))
            sess.run(restores)

        self.train = train
        self.save = save
        self.load = load
        self.train_model = train_model
        self.step_model = step_model
        self.step = step_model.step
        self.value = step_model.value
        self.initial_state = step_model.initial_state
        tf.global_variables_initializer().run(session=sess)


def learn(policy, env, seed, total_timesteps=int(40e6), gamma=0.99, log_interval=1, nprocs=32, n_steps=20,
          ent_coef=0.01, vf_coef=0.5, vf_fisher_coef=1.0, learning_rate=0.25, max_grad_norm=0.5,
          kfac_clip=0.001, save_interval=None, lr_schedule='linear'):
    """
    Traines an ACKTR model.

    :param policy: (Object) The policy model to use (MLP, CNN, LSTM, ...)
    :param env: (Gym environment) The environment to learn from
    :param seed: (int) The initial seed for training
    :param total_timesteps: (int) The total number of samples
    :param gamma: (float) Discount factor
    :param log_interval: (int) The number of timesteps before logging.
    :param nprocs: (int) The number of threads for TensorFlow operations
    :param n_steps: (int) The number of steps to run for each environment
    :param ent_coef: (float) The weight for the entropic loss
    :param vf_coef: (float) The weight for the loss on the value function
    :param vf_fisher_coef: (float) The weight for the fisher loss on the value function
    :param learning_rate: (float) The learning rate
    :param max_grad_norm: (float) The maximum value for the gradient clipping
    :param kfac_clip: (float) gradient clipping for Kullback leiber
    :param save_interval: (int) The number of timesteps before saving.
    :param lr_schedule: (str) The type of scheduler for the learning rate update ('linear', 'constant',
                                 'double_linear_con', 'middle_drop' or 'double_middle_drop')
    """
    set_global_seeds(seed)

    n_envs = env.num_envs
    ob_space = env.observation_space
    ac_space = env.action_space
    make_model = lambda: Model(policy, ob_space, ac_space, n_envs, total_timesteps, nprocs=nprocs, n_steps=n_steps,
                               ent_coef=ent_coef, vf_coef=vf_coef, vf_fisher_coef=vf_fisher_coef,
                               learning_rate=learning_rate,
                               max_grad_norm=max_grad_norm, kfac_clip=kfac_clip, lr_schedule=lr_schedule)
    if save_interval and logger.get_dir():
        import cloudpickle
        with open(os.path.join(logger.get_dir(), 'make_model.pkl'), 'wb') as file_handler:
            file_handler.write(cloudpickle.dumps(make_model))
    model = make_model()

    runner = A2CRunner(env, model, n_steps=n_steps, gamma=gamma)
    n_batch = n_envs * n_steps
    t_start = time.time()
    coord = tf.train.Coordinator()
    enqueue_threads = model.q_runner.create_threads(model.sess, coord=coord, start=True)
    for update in range(1, total_timesteps // n_batch + 1):
        obs, states, rewards, masks, actions, values = runner.run()
        policy_loss, value_loss, policy_entropy = model.train(obs, states, rewards, masks, actions, values)
        model.old_obs = obs
        n_seconds = time.time() - t_start
        fps = int((update * n_batch) / n_seconds)
        if update % log_interval == 0 or update == 1:
            explained_var = explained_variance(values, rewards)
            logger.record_tabular("nupdates", update)
            logger.record_tabular("total_timesteps", update * n_batch)
            logger.record_tabular("fps", fps)
            logger.record_tabular("policy_entropy", float(policy_entropy))
            logger.record_tabular("policy_loss", float(policy_loss))
            logger.record_tabular("value_loss", float(value_loss))
            logger.record_tabular("explained_variance", float(explained_var))
            logger.dump_tabular()

        if save_interval and (update % save_interval == 0 or update == 1) and logger.get_dir():
            savepath = os.path.join(logger.get_dir(), 'checkpoint%.5i' % update)
            print('Saving to', savepath)
            model.save(savepath)
    coord.request_stop()
    coord.join(enqueue_threads)
    env.close()
