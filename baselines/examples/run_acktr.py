import os
import time
import numpy as np
import tensorflow as tf
from utils import logger
from functools import partial
from policies.cnn_policy import CnnPolicy
from utils.misc import set_global_seeds
from algos.acktr_disc import AcktrDiscrete
from algos.acktr_cont import fit as cont_fit
from policies.mlp_gaussian import GaussianMlp
from policies.netvalue import NetValueFunction
from common.vec_env.vec_frame_stack import VecFrameStack
from common.vec_env.environment import AbstractEnvRunner
from utils.math_util import explained_variance, discount_with_dones
from utils.cmd import (arg_parser, make_atari_env, atari_arg_parser,
                       make_mujoco_env, mujoco_arg_parser)


class Environment(AbstractEnvRunner):

    def __init__(self, env, model, nsteps=5, gamma=0.99):

        super().__init__(env=env, model=model, nsteps=nsteps)
        self.gamma = gamma

    def run(self):
        mb_obs, mb_rewards, mb_actions, mb_values, mb_dones = [], [], [], [], []
        mb_states = self.states
        for n in range(self.nsteps):
            # actions, values, states, _ = self.model.step(
            #     self.obs, self.states, self.dones
            # )
            actions, values, states, _ = self.model.step_model.step(
                self.obs, self.states, self.dones
            )
            mb_obs.append(np.copy(self.obs))
            mb_actions.append(actions)
            mb_values.append(values)
            mb_dones.append(self.dones)
            obs, rewards, dones, _ = self.env.step(actions)
            self.states = states
            self.dones = dones
            for n, done in enumerate(dones):
                if done:
                    self.obs[n] = self.obs[n]*0

            self.obs = obs
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
        last_values = self.model.step_model.value(
            self.obs, self.states, self.dones
        ).tolist()
        # discount/bootstrap off value fn
        for n, (rewards, dones, value) in enumerate(zip(mb_rewards, mb_dones,
                                                        last_values)):
            rewards = rewards.tolist()
            dones = dones.tolist()
            if dones[-1] == 0:
                rewards = discount_with_dones(
                    rewards + [value],
                    dones + [0],
                    self.gamma)[:-1]
            else:
                rewards = discount_with_dones(rewards, dones, self.gamma)

            mb_rewards[n] = rewards

        mb_rewards = mb_rewards.flatten()
        mb_actions = mb_actions.flatten()
        mb_values = mb_values.flatten()
        mb_masks = mb_masks.flatten()
        return mb_obs, mb_states, mb_rewards, mb_masks, mb_actions, mb_values


def fit(
        policy,
        env,
        seed,
        total_timesteps=int(40e6),
        gamma=0.99,
        log_interval=1,
        nprocs=32,
        nsteps=20,
        ent_coef=0.01,
        vf_coef=0.5,
        vf_fisher_coef=1.0,
        lr=0.25,
        max_grad_norm=0.5,
        kfac_clip=0.001,
        save_interval=None,
        lrschedule='linear'
):
    tf.reset_default_graph()
    set_global_seeds(seed)

    nenvs = env.num_envs
    ob_space = env.observation_space
    ac_space = env.action_space
    model = AcktrDiscrete(
        policy,
        ob_space,
        ac_space,
        nenvs,
        total_timesteps,
        nsteps=nsteps,
        ent_coef=ent_coef,
        vf_coef=vf_fisher_coef,
        lr=lr,
        max_grad_norm=max_grad_norm,
        kfac_clip=kfac_clip,
        lrschedule=lrschedule
    )
    # if save_interval and logger.get_dir():
    #     import cloudpickle
    #     with open(os.path.join(logger.get_dir(), 'make_model.pkl'), 'wb') as fh:
    #         fh.write(cloudpickle.dumps(make_model))
    # model = make_model()

    runner = Environment(env, model, nsteps=nsteps, gamma=gamma)
    nbatch = nenvs*nsteps
    tstart = time.time()
    coord = tf.train.Coordinator()
    enqueue_threads = model.q_runner.create_threads(
        model.sess, coord=coord, start=True
    )
    for update in range(1, total_timesteps//nbatch+1):
        obs, states, rewards, masks, actions, values = runner.run()
        policy_loss, value_loss, policy_entropy = model.train(
            obs, states, rewards, masks, actions, values
        )
        model.old_obs = obs
        nseconds = time.time()-tstart
        fps = int((update*nbatch)/nseconds)
        if update % log_interval == 0 or update == 1:
            ev = explained_variance(values, rewards)
            logger.record_tabular("nupdates", update)
            logger.record_tabular("total_timesteps", update*nbatch)
            logger.record_tabular("fps", fps)
            logger.record_tabular("policy_entropy", float(policy_entropy))
            logger.record_tabular("policy_loss", float(policy_loss))
            logger.record_tabular("value_loss", float(value_loss))
            logger.record_tabular("explained_variance", float(ev))
            logger.dump_tabular()

        if save_interval and (update % save_interval == 0 or update == 1) \
           and logger.get_dir():
            savepath = os.path.join(logger.get_dir(), 'checkpoint%.5i' % update)
            print('Saving to', savepath)
            model.save(savepath)
    coord.request_stop()
    coord.join(enqueue_threads)
    env.close()


def main():
    parser = arg_parser()
    parser.add_argument('--environ', help='environment choice',
                        choices=['atari', 'mujoco'], default='atari')
    args = parser.parse_args()
    num_cpu = 16

    if args.environ == 'atari':
        # for atari
        args = atari_arg_parser().parse_known_args()[0]
        env = VecFrameStack(make_atari_env(args.env, num_cpu, args.seed), 4)
        policy_fn = partial(CnnPolicy, one_dim_bias=True)
        # policy_fn = Convnet
        fit(
            policy_fn,
            env,
            args.seed,
            total_timesteps=int(args.num_timesteps * 1.1),
        )
        env.close()
    elif args.environ == 'mujoco':
            # for mujoco
        args = mujoco_arg_parser().parse_known_args()[0]
        env = make_mujoco_env(args.env, args.seed)
        with tf.Session(config=tf.ConfigProto()):
            ob_dim = env.observation_space.shape[0]
            ac_dim = env.action_space.shape[0]
            with tf.variable_scope("vf"):
                vf = NetValueFunction(ob_dim, ac_dim)
            with tf.variable_scope("pi"):
                policy = GaussianMlp(ob_dim, ac_dim)

            cont_fit(
                env,
                policy=policy,
                vf=vf,
                gamma=0.99,
                lam=0.97,
                timesteps_per_batch=2500,
                desired_kl=0.002,
                num_timesteps=args.num_timesteps,
                animate=False
            )
            env.close()
    else:
        raise ValueError("Wrong environment. Please choose among atari "
                         "or mujoco!")


if __name__ == '__main__':
    main()
