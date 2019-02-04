import gym.spaces
import time
import numpy as np
import tensorflow as tf
from utils import logger
from algos.a2c import A2C
from policies.lstm import Lstm
from policies.lnlstm import LnLstm
from policies.convnet import Convnet
from utils.misc import set_global_seeds
from utils.math import explained_variance, discount_with_dones
from utils.cmd import make_atari_env, atari_arg_parser
from common.vec_env.vec_frame_stack import VecFrameStack
from common.vec_env.environment import AbstractEnvRunner


class Environment(AbstractEnvRunner):

    def __init__(self, env, model, nsteps=5, gamma=0.99):

        super().__init__(env=env, model=model, nsteps=nsteps)
        self.gamma = gamma

    def run(self, session):
        mb_obs, mb_rewards, mb_actions, mb_values, mb_dones = [], [], [], [], []
        mb_states = self.states
        for n in range(self.nsteps):
            # actions, values, states, _ = self.model.step(
            #     self.obs, self.states, self.dones
            # )
            actions, values, states, _ = self.model.step_model.step(
                self.obs, session=session
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
            self.obs, self.states, self.dones, session=session
        ).tolist()
        # discount/bootstrap off value fn
        for n, (rewards, dones, value) in enumerate(zip(mb_rewards,
                                                        mb_dones,
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
        nsteps=5,
        total_timesteps=int(80e6),
        vf_coef=0.5,
        ent_coef=0.01,
        max_grad_norm=0.5,
        lr=7e-4,
        lrschedule='linear',
        epsilon=1e-5,
        alpha=0.99,
        gamma=0.99,
        log_interval=100
):

    set_global_seeds(seed)

    model = A2C(
        policy=policy,
        observation_space=env.observation_space,
        action_space=env.action_space,
        nenvs=env.num_envs,
        nsteps=nsteps,
        ent_coef=ent_coef,
        vf_coef=vf_coef,
        max_grad_norm=max_grad_norm,
        lr=lr,
        alpha=alpha,
        epsilon=epsilon,
        total_timesteps=total_timesteps,
        lrschedule=lrschedule
    )
    session = model.init_session()
    tf.global_variables_initializer().run(session=session)
    env_runner = Environment(env, model, nsteps=nsteps, gamma=gamma)

    nbatch = env.num_envs * nsteps
    tstart = time.time()
    writer = tf.summary.FileWriter('output', session.graph)
    for update in range(1, total_timesteps//nbatch+1):
        tf.reset_default_graph()
        obs, states, rewards, masks, actions, values = env_runner.run(session)
        policy_loss, value_loss, policy_entropy = model.predict(
            observations=obs,
            states=states,
            rewards=rewards,
            masks=masks,
            actions=actions,
            values=values,
            session=session
        )
        nseconds = time.time() - tstart
        fps = int((update*nbatch) / nseconds)
        if update % log_interval == 0 or update == 1:
            ev = explained_variance(values, rewards)
            logger.record_tabular("nupdates", update)
            logger.record_tabular("total_timesteps", update*nbatch)
            logger.record_tabular("fps", fps)
            logger.record_tabular("policy_entropy", float(policy_entropy))
            logger.record_tabular("value_loss", float(value_loss))
            logger.record_tabular("explained_variance", float(ev))
            logger.dump_tabular()
    env.close()
    writer.close()
    session.close()


def main():
    parser = atari_arg_parser()
    parser.add_argument(
        '--policy',
        help='Policy architecture',
        choices=['cnn', 'lstm', 'lnlstm'],
        default='cnn'
    )

    parser.add_argument(
        '--lrschedule',
        help='Learning rate schedule',
        choices=['constant', 'linear'],
        default='constant'
    )

    args = parser.parse_args()
    logger.configure()

    if args.policy == 'cnn':
        policy_fn = Convnet
    elif args.policy == 'lstm':
        policy_fn = Lstm
    elif args.policy == 'lnlstm':
        policy_fn = LnLstm

    num_env = 16
    env = VecFrameStack(make_atari_env(args.env, num_env, args.seed), 4)
    fit(
        policy_fn,
        env,
        args.seed,
        total_timesteps=int(args.num_timesteps * 1.1),
        lrschedule=args.lrschedule
    )
    env.close()


if __name__ == '__main__':
    main()
