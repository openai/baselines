"""
Discrete acktr
"""

import time
import joblib
import os

import tensorflow as tf
import cloudpickle

from baselines import logger
from baselines.common import set_global_seeds, explained_variance, BaseRLModel
from baselines.a2c.a2c import A2CRunner
from baselines.a2c.utils import Scheduler, find_trainable_variables, calc_entropy, mse, make_path
from baselines.acktr import kfac


class ACKTR(BaseRLModel):
    def __init__(self, policy, env, gamma=0.99, total_timesteps=int(40e6), nprocs=1, n_steps=20,
                 ent_coef=0.01, vf_coef=0.25, vf_fisher_coef=1.0, learning_rate=0.25, max_grad_norm=0.5,
                 kfac_clip=0.001, lr_schedule='linear', _init_setup_model=True):
        """
        The ACKTR (Actor Critic using Kronecker-Factored Trust Region) model class, https://arxiv.org/abs/1708.05144

        :param policy: (Object) The policy model to use (MLP, CNN, LSTM, ...)
        :param env: (Gym environment) The environment to learn from
        :param gamma: (float) Discount factor
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

        :param _init_setup_model: (bool) Whether or not to build the network at the creation of the instance
        """
        super(ACKTR, self).__init__()
        config = tf.ConfigProto(allow_soft_placement=True,
                                intra_op_parallelism_threads=nprocs,
                                inter_op_parallelism_threads=nprocs)
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config)

        self.n_envs = env.num_envs
        self.ob_space = env.observation_space
        self.ac_space = env.action_space
        self.n_batch = self.n_envs * n_steps
        self.env = env
        self.n_steps = n_steps
        self.gamma = gamma
        self.total_timesteps = total_timesteps
        self.policy = policy
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.vf_fisher_coef = vf_fisher_coef
        self.kfac_clip = kfac_clip
        self.max_grad_norm = max_grad_norm
        self.learning_rate_init = learning_rate
        self.lr_schedule = lr_schedule
        self.nprocs = nprocs

        self.action_ph = None
        self.advs_ph = None
        self.rewards_ph = None
        self.pg_lr_ph = None
        self.model = None
        self.model2 = None
        self.logits = None
        self.entropy = None
        self.pg_loss = None
        self.vf_loss = None
        self.pg_fisher = None
        self.vf_fisher = None
        self.joint_fisher = None
        self.params = None
        self.grads_check = None
        self.optim = None
        self.train_op = None
        self.q_runner = None
        self.learning_rate = None
        self.train_model = None
        self.step_model = None
        self.step = None
        self.value = None
        self.initial_state = None

        if _init_setup_model:
            self.setup_model()

    def setup_model(self):
        self.action_ph = action_ph = tf.placeholder(tf.int32, [self.n_batch])
        self.advs_ph = advs_ph = tf.placeholder(tf.float32, [self.n_batch])
        self.rewards_ph = rewards_ph = tf.placeholder(tf.float32, [self.n_batch])
        self.pg_lr_ph = pg_lr_ph = tf.placeholder(tf.float32, [])

        self.model = step_model = self.policy(self.sess, self.ob_space, self.ac_space, self.n_envs, 1, reuse=False)
        self.model2 = train_model = self.policy(self.sess, self.ob_space, self.ac_space, self.n_envs * self.n_steps,
                                                self.n_steps, reuse=True)

        logpac = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=train_model.policy, labels=action_ph)
        self.logits = train_model.policy

        # training loss
        pg_loss = tf.reduce_mean(advs_ph * logpac)
        self.entropy = entropy = tf.reduce_mean(calc_entropy(train_model.policy))
        self.pg_loss = pg_loss = pg_loss - self.ent_coef * entropy
        self.vf_loss = vf_loss = mse(tf.squeeze(train_model.value_fn), rewards_ph)
        train_loss = pg_loss + self.vf_coef * vf_loss

        # Fisher loss construction
        self.pg_fisher = pg_fisher_loss = -tf.reduce_mean(logpac)
        sample_net = train_model.value_fn + tf.random_normal(tf.shape(train_model.value_fn))
        self.vf_fisher = vf_fisher_loss = - self.vf_fisher_coef * tf.reduce_mean(
            tf.pow(train_model.value_fn - tf.stop_gradient(sample_net), 2))
        self.joint_fisher = pg_fisher_loss + vf_fisher_loss

        self.params = params = find_trainable_variables("model")

        self.grads_check = grads = tf.gradients(train_loss, params)

        with tf.device('/gpu:0'):
            self.optim = optim = kfac.KfacOptimizer(learning_rate=pg_lr_ph, clip_kl=self.kfac_clip,
                                                    momentum=0.9, kfac_update=1, epsilon=0.01,
                                                    stats_decay=0.99, async=1, cold_iter=10,
                                                    max_grad_norm=self.max_grad_norm)

            optim.compute_and_apply_stats(self.joint_fisher, var_list=params)
            self.train_op, self.q_runner = optim.apply_gradients(list(zip(grads, params)))

        self.learning_rate = Scheduler(initial_value=self.learning_rate_init, n_values=self.total_timesteps,
                                       schedule=self.lr_schedule)

        self.train_model = train_model
        self.step_model = step_model
        self.step = step_model.step
        self.value = step_model.value
        self.initial_state = step_model.initial_state
        tf.global_variables_initializer().run(session=self.sess)

    def _train_step(self, obs, states, rewards, masks, actions, values):
        """
        applies a training step to the model

        :param obs: ([float]) The input observations
        :param states: ([float]) The states (used for reccurent policies)
        :param rewards: ([float]) The rewards from the environment
        :param masks: ([bool]) Whether or not the episode is over (used for reccurent policies)
        :param actions: ([float]) The actions taken
        :param values: ([float]) The logits values
        :return: (float, float, float) policy loss, value loss, policy entropy
        """
        advs = rewards - values
        cur_lr = None
        for _ in range(len(obs)):
            cur_lr = self.learning_rate.value()
        assert cur_lr is not None, "Error: the observation input array cannon be empty"

        td_map = {self.train_model.obs_ph: obs, self.action_ph: actions, self.advs_ph: advs, self.rewards_ph: rewards,
                  self.pg_lr_ph: cur_lr}
        if states is not None:
            td_map[self.train_model.states_ph] = states
            td_map[self.train_model.masks_ph] = masks

        policy_loss, value_loss, policy_entropy, _ = self.sess.run(
            [self.pg_loss, self.vf_loss, self.entropy, self.train_op],
            td_map
        )
        return policy_loss, value_loss, policy_entropy

    def learn(self, callback=None, seed=None, log_interval=100):
        if seed is not None:
            set_global_seeds(seed)

        runner = A2CRunner(self.env, self, n_steps=self.n_steps, gamma=self.gamma)

        t_start = time.time()
        coord = tf.train.Coordinator()
        enqueue_threads = self.q_runner.create_threads(self.sess, coord=coord, start=True)
        for update in range(1, self.total_timesteps // self.n_batch + 1):
            obs, states, rewards, masks, actions, values = runner.run()
            policy_loss, value_loss, policy_entropy = self._train_step(obs, states, rewards, masks, actions, values)
            n_seconds = time.time() - t_start
            fps = int((update * self.n_batch) / n_seconds)

            if callback is not None:
                callback(locals(), globals())

            if update % log_interval == 0 or update == 1:
                explained_var = explained_variance(values, rewards)
                logger.record_tabular("nupdates", update)
                logger.record_tabular("total_timesteps", update * self.n_batch)
                logger.record_tabular("fps", fps)
                logger.record_tabular("policy_entropy", float(policy_entropy))
                logger.record_tabular("policy_loss", float(policy_loss))
                logger.record_tabular("value_loss", float(value_loss))
                logger.record_tabular("explained_variance", float(explained_var))
                logger.dump_tabular()

        coord.request_stop()
        coord.join(enqueue_threads)
        return self

    def save(self, save_path):
        data = {
            "gamma": self.gamma,
            "nprocs": self.nprocs,
            "n_steps": self.n_steps,
            "vf_coef": self.vf_coef,
            "ent_coef": self.ent_coef,
            "vf_fisher_coef": self.vf_fisher_coef,
            "max_grad_norm": self.max_grad_norm,
            "learning_rate_init": self.learning_rate_init,
            "kfac_clip": self.kfac_clip,
            "lr_schedule": self.lr_schedule,
            "policy": self.policy,
            "ob_space": self.ob_space,
            "ac_space": self.ac_space
        }

        with open(".".join(save_path.split('.')[:-1]) + "_class.pkl", "wb") as file:
            cloudpickle.dump(data, file)

        parameters = self.sess.run(self.params)
        make_path(os.path.dirname(save_path))
        joblib.dump(parameters, save_path)

    @classmethod
    def load(cls, load_path, env, **kwargs):
        if "learning_rate" in kwargs:
            kwargs["learning_rate_init"] = kwargs["learning_rate"]

        with open(".".join(load_path.split('.')[:-1]) + "_class.pkl", "rb") as file:
            data = cloudpickle.load(file)

        assert data["ob_space"] == env.observation_space, \
            "Error: the environment passed must have at least the same observation space as the model was trained on."
        assert data["ac_space"] == env.action_space, \
            "Error: the environment passed must have at least the same action space as the model was trained on."

        model = cls(policy=data["policy"], env=env, _init_setup_model=False)
        model.__dict__.update(data)
        model.__dict__.update(kwargs)
        model.setup_model()

        loaded_params = joblib.load(load_path)
        restores = []
        for param, loaded_p in zip(model.params, loaded_params):
            restores.append(param.assign(loaded_p))
        model.sess.run(restores)

        return model
