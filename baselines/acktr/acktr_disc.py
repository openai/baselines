import joblib
import numpy as np
import os.path as osp
import tensorflow as tf
import time
from baselines import logger
from baselines.acktr import kfac
from baselines.acktr.utils import cat_entropy, mse
from baselines.acktr.utils import discount_with_dones
from baselines.acktr.utils import Scheduler, find_trainable_variables
from baselines.common import set_global_seeds, explained_variance


class Model(object):

    def __init__(
            self, policy, ob_space, ac_space,
            nenvs, nprocs=32, nstack=4, nsteps=20, total_timesteps=int(80e6),
            ent_coef=0.01, vf_coef=0.5, vf_fisher_coef=1.0,
            lr=0.25, max_grad_norm=0.5,
            kfac_clip=0.001, lrschedule='linear'):
        config = tf.ConfigProto(
            allow_soft_placement=True,
            intra_op_parallelism_threads=nprocs,
            inter_op_parallelism_threads=nprocs)
        config.gpu_options.allow_growth = True
        self.sess = sess = tf.Session(config=config)
        self.act_model = act_model = policy(
            sess, ob_space, ac_space, nenvs, 1, nstack, reuse=False)
        self.train_model = train_model = policy(
            sess, ob_space, ac_space, nenvs, nsteps, nstack, reuse=True)

        nact = ac_space.n
        nbatch = nenvs * nsteps
        A = tf.placeholder(tf.int32, [nbatch])
        ADV = tf.placeholder(tf.float32, [nbatch])
        R = tf.placeholder(tf.float32, [nbatch])
        PG_LR = tf.placeholder(tf.float32, [])
        VF_LR = tf.placeholder(tf.float32, [])

        # define training loss
        neglogpac = tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=train_model.pi, labels=A)
        entropy = tf.reduce_mean(cat_entropy(train_model.pi))

        pg_loss = tf.reduce_mean(ADV * neglogpac)
        vf_loss = tf.reduce_mean(mse(tf.squeeze(train_model.vf), R))
        loss = pg_loss - entropy * ent_coef + vf_loss * vf_coef

        self.logits = logits = train_model.pi

        # Fisher loss construction
        self.pg_fisher = pg_fisher_loss = -tf.reduce_mean(neglogpac)
        sample_net = train_model.vf + \
            tf.random_normal(tf.shape(train_model.vf))
        self.vf_fisher = vf_fisher_loss = - vf_fisher_coef * \
            tf.reduce_mean(
                tf.pow(train_model.vf - tf.stop_gradient(sample_net), 2))
        self.joint_fisher = joint_fisher_loss = pg_fisher_loss + vf_fisher_loss

        self.params = params = find_trainable_variables("model")

        self.grads_check = grads = tf.gradients(loss, params)

        with tf.device('/gpu:0'):
            self.optim = optim = kfac.KfacOptimizer(
                learning_rate=PG_LR, clip_kl=kfac_clip,
                momentum=0.9, kfac_update=1, epsilon=0.01,
                stats_decay=0.99, async=1, cold_iter=10, max_grad_norm=max_grad_norm)

            update_stats_op = optim.compute_and_apply_stats(
                joint_fisher_loss, var_list=params)
            train_op, q_runner = optim.apply_gradients(
                list(zip(grads, params)))
        self.q_runner = q_runner
        self.lr = Scheduler(v=lr, nvalues=total_timesteps, schedule=lrschedule)

        def train(obs, states, rewards, masks, actions, values):
            advs = rewards - values
            for step in range(len(obs)):
                cur_lr = self.lr.value()

            td_map = {train_model.X: obs, A: actions,
                      ADV: advs, R: rewards, PG_LR: cur_lr}
            if states != []:
                td_map[train_model.S] = states
                td_map[train_model.M] = masks

            policy_loss, value_loss, policy_entropy, _ = sess.run(
                [pg_loss, vf_loss, entropy, train_op],
                td_map
            )
            return policy_loss, value_loss, policy_entropy

        def save(save_path):
            ps = sess.run(params)
            joblib.dump(ps, save_path)

        def load(load_path):
            loaded_params = joblib.load(load_path)
            restores = []
            for p, loaded_p in zip(params, loaded_params):
                restores.append(p.assign(loaded_p))
            sess.run(restores)

        self.train = train
        self.save = save
        self.load = load
        self.train_model = train_model
        self.act_model = act_model
        self.step = act_model.step
        self.value = act_model.value
        self.initial_state = act_model.initial_state
        tf.global_variables_initializer().run(session=sess)


class Runner(object):

    def __init__(self, env, model, nsteps, nstack, gamma):
        self.env = env
        self.act_model = model
        nh, nw, nc = env.observation_space.shape
        nenv = env.num_envs
        self.batch_ob_shape = (nenv * nsteps, nh, nw, nc * nstack)
        self.obs = np.zeros((nenv, nh, nw, nc * nstack), dtype=np.uint8)
        obs = env.reset()
        self.update_obs(obs)
        self.gamma = gamma
        self.nsteps = nsteps
        self.states = model.initial_state
        self.dones = [False for _ in range(nenv)]

    def update_obs(self, obs):
        self.obs = np.roll(self.obs, shift=-1, axis=3)
        self.obs[:, :, :, -1] = obs[:, :, :, 0]

    def run(self):
        mb_obs, mb_rewards, mb_actions, mb_values, mb_dones = [], [], [], [], []
        mb_states = self.states
        for n in range(self.nsteps):
            actions, values, states = self.act_model.step(
                self.obs, self.states, self.dones)
            mb_obs.append(np.copy(self.obs))
            mb_actions.append(actions)
            mb_values.append(values)
            mb_dones.append(self.dones)
            obs, rewards, dones, _ = self.env.step(actions)
            self.states = states
            self.dones = dones
            for n, done in enumerate(dones):
                if done:
                    self.obs[n] = self.obs[n] * 0
            self.update_obs(obs)
            mb_rewards.append(rewards)
        mb_dones.append(self.dones)
        # batch of steps to batch of rollouts
        mb_obs = np.asarray(mb_obs, dtype=np.uint8).swapaxes(
            1, 0).reshape(self.batch_ob_shape)
        mb_rewards = np.asarray(mb_rewards, dtype=np.float32).swapaxes(1, 0)
        mb_actions = np.asarray(mb_actions, dtype=np.int32).swapaxes(1, 0)
        mb_values = np.asarray(mb_values, dtype=np.float32).swapaxes(1, 0)
        mb_dones = np.asarray(mb_dones, dtype=np.bool).swapaxes(1, 0)
        mb_masks = mb_dones[:, :-1]
        mb_dones = mb_dones[:, 1:]
        last_values = self.act_model.value(
            self.obs, self.states, self.dones).tolist()
        # discount/bootstrap off value fn
        for n, (rewards, dones, value) in enumerate(zip(mb_rewards, mb_dones, last_values)):
            rewards = rewards.tolist()
            dones = dones.tolist()
            if dones[-1] == 0:
                rewards = discount_with_dones(
                    rewards + [value], dones + [0], self.gamma)[:-1]
            else:
                rewards = discount_with_dones(rewards, dones, self.gamma)
            mb_rewards[n] = rewards
        mb_rewards = mb_rewards.flatten()
        mb_actions = mb_actions.flatten()
        mb_values = mb_values.flatten()
        mb_masks = mb_masks.flatten()
        return mb_obs, mb_states, mb_rewards, mb_masks, mb_actions, mb_values


def learn(policy, env, seed, total_timesteps=int(40e6), gamma=0.99, log_interval=1, nprocs=32, nsteps=20,
          nstack=4, ent_coef=0.01, vf_coef=0.5, vf_fisher_coef=1.0, lr=0.25, max_grad_norm=0.5,
          kfac_clip=0.001, save_interval=None, lrschedule='linear'):
    tf.reset_default_graph()
    set_global_seeds(seed)

    nenvs = env.num_envs
    ob_space = env.observation_space
    ac_space = env.action_space

    def make_model(): return Model(
        policy, ob_space, ac_space,
        nenvs=nenvs, nprocs=nprocs, nstack=nstack, nsteps=nsteps,
        total_timesteps=total_timesteps,
        ent_coef=ent_coef, vf_coef=vf_coef, vf_fisher_coef=vf_fisher_coef,
        lr=lr, max_grad_norm=max_grad_norm, kfac_clip=kfac_clip,
        lrschedule=lrschedule)
    if save_interval and logger.get_dir():
        import cloudpickle
        with open(osp.join(logger.get_dir(), 'make_model.pkl'), 'wb') as fh:
            fh.write(cloudpickle.dumps(make_model))
    model = make_model()

    runner = Runner(env, model, nsteps=nsteps, nstack=nstack, gamma=gamma)
    nbatch = nenvs * nsteps
    tstart = time.time()
    coord = tf.train.Coordinator()
    enqueue_threads = model.q_runner.create_threads(
        model.sess, coord=coord, start=True)
    for update in range(1, total_timesteps // nbatch + 1):
        obs, states, rewards, masks, actions, values = runner.run()
        policy_loss, value_loss, policy_entropy = model.train(
            obs, states, rewards, masks, actions, values)
        model.old_obs = obs
        nseconds = time.time() - tstart
        fps = int((update * nbatch) / nseconds)
        if update % log_interval == 0 or update == 1:
            ev = explained_variance(values, rewards)
            logger.record_tabular("nupdates", update)
            logger.record_tabular("total_timesteps", update * nbatch)
            logger.record_tabular("fps", fps)
            logger.record_tabular("policy_entropy", float(policy_entropy))
            logger.record_tabular("policy_loss", float(policy_loss))
            logger.record_tabular("value_loss", float(value_loss))
            logger.record_tabular("explained_variance", float(ev))
            logger.dump_tabular()

        if save_interval and (update % save_interval == 0 or update == 1) and logger.get_dir():
            savepath = osp.join(logger.get_dir(), 'checkpoint%.5i' % update)
            print('Saving to', savepath)
            model.save(savepath)
    coord.request_stop()
    coord.join(enqueue_threads)
    env.close()
