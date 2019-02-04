import gym.spaces
import time
import numpy as np
import tensorflow as tf
from utils import logger
from algos.acer import Acer, AgentEnv
from policies.agent import Agent
from dstruct.buffers import Buffer
from utils.misc import set_global_seeds
from policies.acer_lstm import AcerLstm
from policies.acer_convnet import AcerConvnet
from utils.cmd import make_atari_env, atari_arg_parser
from common.vec_env.environment import AbstractEnvRunner


class Environment(AbstractEnvRunner):
    def __init__(self, env, model, nsteps, nstack):
        super().__init__(env=env, model=model, nsteps=nsteps)
        self.nstack = nstack
        nh, nw, nc = env.observation_space.shape
        self.nc = nc  # nc = 1 for atari, but just in case
        self.nenv = nenv = env.num_envs
        self.nact = env.action_space.n
        self.nbatch = nenv * nsteps
        self.batch_ob_shape = (nenv*(nsteps+1), nh, nw, nc*nstack)
        self.obs = np.zeros((nenv, nh, nw, nc * nstack), dtype=np.uint8)
        obs = env.reset()
        self.update_obs(obs)

    def update_obs(self, obs, dones=None):
        if dones is not None:
            self.obs *= (1 - dones.astype(np.uint8))[:, None, None, None]
        self.obs = np.roll(self.obs, shift=-self.nc, axis=3)
        self.obs[:, :, :, -self.nc:] = obs[:, :, :, :]

    def run(self):
        enc_obs = np.split(self.obs, self.nstack, axis=3)  # so now list of obs steps
        mb_obs, mb_actions, mb_mus, mb_dones, mb_rewards = [], [], [], [], []
        for _ in range(self.nsteps):
            actions, mus, states = self.model.step_model.step(
                self.obs, state=self.states, mask=self.dones
            )
            mb_obs.append(np.copy(self.obs))
            mb_actions.append(actions)
            mb_mus.append(mus)
            mb_dones.append(self.dones)
            obs, rewards, dones, _ = self.env.step(actions)
            # states information for statefull models like LSTM
            self.states = states
            self.dones = dones
            self.update_obs(obs, dones)
            mb_rewards.append(rewards)
            enc_obs.append(obs)
        mb_obs.append(np.copy(self.obs))
        mb_dones.append(self.dones)

        enc_obs = np.asarray(enc_obs, dtype=np.uint8).swapaxes(1, 0)
        mb_obs = np.asarray(mb_obs, dtype=np.uint8).swapaxes(1, 0)
        mb_actions = np.asarray(mb_actions, dtype=np.int32).swapaxes(1, 0)
        mb_rewards = np.asarray(mb_rewards, dtype=np.float32).swapaxes(1, 0)
        mb_mus = np.asarray(mb_mus, dtype=np.float32).swapaxes(1, 0)

        mb_dones = np.asarray(mb_dones, dtype=np.bool).swapaxes(1, 0)

        # Used for statefull models like LSTM's to mask state when done
        mb_masks = mb_dones
        # Used for calculating returns. The dones array is now aligned
        # with rewards
        mb_dones = mb_dones[:, 1:]

        # shapes are now [nenv, nsteps, []]
        # When pulling from buffer, arrays will now be reshaped in
        # place, preventing a deep copy.

        return enc_obs, mb_obs, mb_actions, mb_rewards, mb_mus,\
            mb_dones, mb_masks


def fit(
        policy,
        env,
        seed,
        nsteps=20,
        nstack=4,
        total_timesteps=int(80e6),
        q_coef=0.5,
        ent_coef=0.01,
        max_grad_norm=10,
        lr=7e-4,
        lrschedule='linear',
        rprop_epsilon=1e-5,
        rprop_alpha=0.99,
        gamma=0.99,
        log_interval=100,
        buffer_size=50000,
        replay_ratio=4,
        replay_start=10000,
        c=10.0,
        trust_region=True,
        alpha=0.99,
        delta=1
):
    print("Running Acer Simple")
    print(locals())
    tf.reset_default_graph()
    set_global_seeds(seed)

    # num_procs = len(env.remotes)  # HACK
    model = Acer(
        policy=policy,
        observation_space=env.observation_space,
        action_space=env.action_space,
        nenvs=env.num_envs,
        nsteps=nsteps,
        nstack=nstack,
        ent_coef=ent_coef,
        q_coef=q_coef,
        gamma=gamma,
        max_grad_norm=max_grad_norm,
        lr=lr,
        rprop_alpha=rprop_alpha,
        rprop_epsilon=rprop_epsilon,
        total_timesteps=total_timesteps,
        lrschedule=lrschedule,
        c=c,
        trust_region=trust_region,
        alpha=alpha,
        delta=delta
    )

    env_runner = Environment(env=env, model=model, nsteps=nsteps, nstack=nstack)
    if replay_ratio > 0:
        buffer = Buffer(env=env, nsteps=nsteps, nstack=nstack, size=buffer_size)
    else:
        buffer = None
    nbatch = env.num_envs * nsteps
    agent = AgentEnv(env_runner, model, buffer, log_interval)
    agent.tstart = time.time()
    # nbatch samples, 1 on_policy call and multiple off-policy calls
    for agent.steps in range(0, total_timesteps, nbatch):
        agent.call(on_policy=True)
        if replay_ratio > 0 and buffer.has_atleast(replay_start):
            n = np.random.poisson(replay_ratio)
            for _ in range(n):
                agent.call(on_policy=False)  # no simulation steps in this

    env.close()


def main():
    parser = atari_arg_parser()
    parser.add_argument('--policy', help='Policy architecture',
                        choices=['cnn', 'lstm', 'lnlstm'], default='cnn')
    parser.add_argument('--lrschedule', help='Learning rate schedule',
                        choices=['constant', 'linear'], default='constant')
    parser.add_argument('--logdir', help='Directory for logging')
    args = parser.parse_args()
    logger.configure(args.logdir)

    num_cpu = 16
    env = make_atari_env(args.env, num_cpu, args.seed)
    if args.policy == 'cnn':
        policy_fn = AcerConvnet
    elif args.policy == 'lstm':
        policy_fn = AcerLstm
    else:
        print("Policy {} not implemented".format(args.policy))
        return
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
