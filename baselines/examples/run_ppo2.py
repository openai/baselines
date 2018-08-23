import gym.spaces
import os
import time
import numpy as np
from utils import logger
from algos.ppo2 import PPO2
from policies.mlp import Mlp
from collections import deque
from policies.lstm import Lstm
from policies.agent import Agent
from policies.lnlstm import LnLstm
from policies.convnet import Convnet
from utils.math import explained_variance
from common.vec_env.dummy_vec_env import DummyVecEnv
from common.vec_env.vec_normalize import VecNormalize
from common.vec_env.vec_frame_stack import VecFrameStack
from common.vec_env.environment import AbstractEnvRunner
from utils.cmd import (arg_parser, atari_arg_parser, mujoco_arg_parser,
                       make_atari_env, make_mujoco_env)


class Environment(AbstractEnvRunner):

    def __init__(self, *, env, model, nsteps, gamma, lam):
        super(Environment, self).__init__(env=env, model=model, nsteps=nsteps)
        self.lam = lam
        self.gamma = gamma

    def run(self):
        mb_obs, mb_rewards, mb_actions, mb_values, mb_dones, mb_neglogpacs =\
            [], [], [], [], [], []
        mb_states = self.states
        epinfos = []
        for _ in range(self.nsteps):
            actions, values, self.states, neglogpacs = self.model.step(
                observations=self.obs,
                states=self.states,
                dones=self.dones
            )
            mb_obs.append(self.obs.copy())
            mb_actions.append(actions)
            mb_values.append(values)
            mb_neglogpacs.append(neglogpacs)
            mb_dones.append(self.dones)
            self.obs[:], rewards, self.dones, infos = self.env.step(actions)
            for info in infos:
                maybeepinfo = info.get('episode')
                if maybeepinfo:
                    epinfos.append(maybeepinfo)
            mb_rewards.append(rewards)
        #batch of steps to batch of rollouts
        mb_obs = np.asarray(mb_obs, dtype=self.obs.dtype)
        mb_rewards = np.asarray(mb_rewards, dtype=np.float32)
        mb_actions = np.asarray(mb_actions)
        mb_values = np.asarray(mb_values, dtype=np.float32)
        mb_neglogpacs = np.asarray(mb_neglogpacs, dtype=np.float32)
        mb_dones = np.asarray(mb_dones, dtype=np.bool)
        last_values = self.model.value(
            observations=self.obs,
            states=self.states,
            dones=self.dones
        )
        #discount/bootstrap off value fn
        mb_returns = np.zeros_like(mb_rewards)
        mb_advs = np.zeros_like(mb_rewards)
        lastgaelam = 0
        for t in reversed(range(self.nsteps)):
            if t == self.nsteps - 1:
                nextnonterminal = 1.0 - self.dones
                nextvalues = last_values
            else:
                nextnonterminal = 1.0 - mb_dones[t+1]
                nextvalues = mb_values[t+1]
            delta = mb_rewards[t] + self.gamma * nextvalues * nextnonterminal - mb_values[t]
            mb_advs[t] = lastgaelam = delta + self.gamma * self.lam * nextnonterminal * lastgaelam
        mb_returns = mb_advs + mb_values
        return (
            *map(
                sf01, (mb_obs, mb_returns, mb_dones, mb_actions, mb_values,
                       mb_neglogpacs)
            ),
            mb_states,
            epinfos
        )


# obs, returns, masks, actions, values, neglogpacs, states = runner.run()
def sf01(arr):
    """
    swap and then flatten axes 0 and 1
    """
    s = arr.shape
    return arr.swapaxes(0, 1).reshape(s[0] * s[1], *s[2:])


def constfn(val):
    def f(_):
        return val
    return f


def safemean(xs):
    return np.nan if len(xs) == 0 else np.mean(xs)


def fit(
        policy,
        env,
        nsteps,
        total_timesteps,
        ent_coef,
        lr,
        vf_coef=0.5,
        max_grad_norm=0.5,
        gamma=0.99,
        lam=0.95,
        log_interval=10,
        nminibatches=4,
        noptepochs=4,
        cliprange=0.2,
        save_interval=0,
        load_path=None
):

    if isinstance(lr, float):
        lr = constfn(lr)
    else:
        assert callable(lr)
    if isinstance(cliprange, float):
        cliprange = constfn(cliprange)
    else:
        assert callable(cliprange)
    total_timesteps = int(total_timesteps)

    nenvs = env.num_envs
    # nenvs = 8
    ob_space = env.observation_space
    ac_space = env.action_space
    nbatch = nenvs * nsteps
    nbatch_train = nbatch // nminibatches

    model = PPO2(
        policy=policy,
        observation_space=ob_space,
        action_space=ac_space,
        nbatch_act=nenvs,
        nbatch_train=nbatch_train,
        nsteps=nsteps,
        ent_coef=ent_coef,
        vf_coef=vf_coef,
        max_grad_norm=max_grad_norm
    )
    Agent().init_vars()
    # if save_interval and logger.get_dir():
    #     import cloudpickle
    #     with open(os.path.join(logger.get_dir(), 'make_model.pkl'), 'wb') as fh:
    #         fh.write(cloudpickle.dumps(make_model))
    # model = make_model()
    # if load_path is not None:
    #     model.load(load_path)
    runner = Environment(env=env, model=model, nsteps=nsteps, gamma=gamma, lam=lam)

    epinfobuf = deque(maxlen=100)
    tfirststart = time.time()

    nupdates = total_timesteps//nbatch
    for update in range(1, nupdates+1):
        assert nbatch % nminibatches == 0
        nbatch_train = nbatch // nminibatches
        tstart = time.time()
        frac = 1.0 - (update - 1.0) / nupdates
        lrnow = lr(frac)
        cliprangenow = cliprange(frac)
        obs, returns, masks, actions, values, neglogpacs, states, epinfos = runner.run()
        epinfobuf.extend(epinfos)
        mblossvals = []
        if states is None:  # nonrecurrent version
            inds = np.arange(nbatch)
            for _ in range(noptepochs):
                np.random.shuffle(inds)
                for start in range(0, nbatch, nbatch_train):
                    end = start + nbatch_train
                    mbinds = inds[start:end]
                    slices = (
                        arr[mbinds] for arr in (obs,
                                                returns,
                                                masks,
                                                actions,
                                                values,
                                                neglogpacs)
                    )
                    mblossvals.append(model.predict(lrnow, cliprangenow, *slices))
        else:  # recurrent version
            assert nenvs % nminibatches == 0
            envsperbatch = nenvs // nminibatches
            envinds = np.arange(nenvs)
            flatinds = np.arange(nenvs * nsteps).reshape(nenvs, nsteps)
            envsperbatch = nbatch_train // nsteps
            for _ in range(noptepochs):
                np.random.shuffle(envinds)
                for start in range(0, nenvs, envsperbatch):
                    end = start + envsperbatch
                    mbenvinds = envinds[start:end]
                    mbflatinds = flatinds[mbenvinds].ravel()
                    slices = (
                        arr[mbflatinds] for arr in (obs,
                                                    returns,
                                                    masks,
                                                    actions,
                                                    values,
                                                    neglogpacs)
                    )
                    mbstates = states[mbenvinds]
                    mblossvals.append(model.predict(lrnow, cliprangenow, *slices, mbstates))

        lossvals = np.mean(mblossvals, axis=0)
        tnow = time.time()
        fps = int(nbatch / (tnow - tstart))
        if update % log_interval == 0 or update == 1:
            ev = explained_variance(values, returns)
            logger.logkv("serial_timesteps", update*nsteps)
            logger.logkv("nupdates", update)
            logger.logkv("total_timesteps", update*nbatch)
            logger.logkv("fps", fps)
            logger.logkv("explained_variance", float(ev))
            logger.logkv('eprewmean', safemean([epinfo['r'] for epinfo in epinfobuf]))
            logger.logkv('eplenmean', safemean([epinfo['l'] for epinfo in epinfobuf]))
            logger.logkv('time_elapsed', tnow - tfirststart)
            for (lossval, lossname) in zip(lossvals, model.loss_names):
                logger.logkv(lossname, lossval)
            logger.dumpkvs()
        if save_interval and (update % save_interval == 0 or update == 1) and logger.get_dir():
            checkdir = os.path.join(logger.get_dir(), 'checkpoints')
            os.makedirs(checkdir, exist_ok=True)
            savepath = os.path.join(checkdir, '%.5i' % update)
            print('Saving to', savepath)
            model.save(savepath)
    env.close()
    return model


# atari
# def train(env_id, num_timesteps, seed, policy):

#     sess = self.init_session()__enter__()
#     env = VecFrameStack(make_atari_env(env_id, 8, seed), 4)
#     policy = {'cnn' : CnnPolicy,
#               'lstm' : LstmPolicy,
#               'lnlstm' : LnLstmPolicy,
#               'mlp': MlpPolicy}[policy]

#     fit(
#         policy=policy,
#         env=env,
#         nsteps=128,
#         nminibatches=4,
#         lam=0.95,
#         gamma=0.99,
#         noptepochs=4,
#         log_interval=1,
#         ent_coef=.01,
#         lr=lambda f : f * 2.5e-4,
#         cliprange=lambda f : f * 0.1,
#         total_timesteps=int(num_timesteps * 1.1)
#     )

def main():
    parser = arg_parser()
    parser.add_argument('--platform', help='environment choice',
                        choices=['atari', 'mujoco', 'humanoid', 'robotics'],
                        default='atari')
    platform_args, environ_args = parser.parse_known_args()
    platform = platform_args.platform
    logger.configure()

    # atari
    if platform == 'atari':
        parser = atari_arg_parser()
        parser.add_argument('--policy', help='Policy architecture',
                            choices=['cnn', 'lstm', 'lnlstm', 'mlp'],
                            default='cnn')
        args = parser.parse_known_args()[0]

        # fit(
        #     args.env,
        #     num_timesteps=args.num_timesteps,
        #     seed=args.seed,
        #     policy=args.policy
        # )
        sess = Agent().init_session().__enter__()
        env = VecFrameStack(make_atari_env(args.env, 8, args.seed), 4)
        policy = {'cnn' : Convnet,
                  'lstm' : Lstm,
                  'lnlstm' : LnLstm,
                  'mlp': Mlp}[args.policy]

        fit(
            policy=policy,
            env=env,
            nsteps=128,
            nminibatches=8,
            lam=0.95,
            gamma=0.99,
            noptepochs=4,
            log_interval=1,
            ent_coef=.01,
            lr=lambda f: f * 2.5e-4,
            cliprange=lambda f: f * 0.1,
            total_timesteps=int(args.num_timesteps * 1.1)
        )

        sess.close()
        env.close()

    # mujoco
    if platform == 'mujoco':
        args = mujoco_arg_parser().parse_known_args()[0]

        sess = Agent().init_session().__enter__()
        from utils.monitor import Monitor

        def make_env():
            env = make_mujoco_env(args.env, args.seed)
            # env = gym.make(env_id)
            env = Monitor(env, logger.get_dir(), allow_early_resets=True)
            return env

        env = DummyVecEnv([make_env])
        env = VecNormalize(env)

        model = fit(
            policy=Mlp,
            env=env,
            nsteps=2048,
            nminibatches=32,
            lam=0.95,
            gamma=0.99,
            noptepochs=10,
            log_interval=1,
            ent_coef=0.0,
            lr=3e-4,
            cliprange=0.2,
            total_timesteps=args.num_timesteps
        )

        # return model, env

        if args.play:
            logger.log("Running trained model")
            obs = np.zeros((env.num_envs,) + env.observation_space.shape)
            obs[:] = env.reset()
            while True:
                actions = model.step(obs)[0]
                obs[:]  = env.step(actions)[0]
                env.render()

        sess.close()
        env.close()


if __name__ == '__main__':
    main()
