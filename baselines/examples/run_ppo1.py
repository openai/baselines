import os
import gym.spaces
from mpi4py import MPI
from utils import logger
from bench import Monitor
from algos.ppo1 import PPOSGD
from policies.model import Model
from policies.ppo1mlp import PPO1Mlp
from policies.ppo1cnn import PPO1Cnn
from utils.misc import set_global_seeds
from common.vec_env.atari_wrappers import make_atari, wrap_deepmind
from utils.cmd import (arg_parser, atari_arg_parser, mujoco_arg_parser,
                       robotics_arg_parser)


def fit(environ, env_id, num_timesteps, seed, model_path=None):
    # atari
    if environ == 'atari':
        rank = MPI.COMM_WORLD.Get_rank()
        sess = Model().single_threaded_session()
        sess.__enter__()
        if rank == 0:
            logger.configure()
        else:
            logger.configure(format_strs=[])
        workerseed = seed + 10000 * MPI.COMM_WORLD.Get_rank() if seed \
            is not None else None
        set_global_seeds(workerseed)
        env = make_atari(env_id)

        def policy_fn(name, ob_space, ac_space):
            return PPO1Cnn(name=name, ob_space=ob_space, ac_space=ac_space)

        env = Monitor(
            env,
            logger.get_dir() and os.path.join(logger.get_dir(), str(rank))
        )
        env.seed(workerseed)

        env = wrap_deepmind(env)
        env.seed(workerseed)

        pi = PPOSGD(
            env,
            policy_fn,
            env.observation_space,
            env.action_space,
            timesteps_per_actorbatch=256,
            clip_param=0.2,
            entcoeff=0.01,
            optim_epochs=4,
            optim_stepsize=1e-3,
            optim_batchsize=64,
            gamma=0.99,
            lam=0.95,
            max_timesteps=int(num_timesteps * 1.1),
            schedule='linear'
            )

        env.close()
        sess.close()
        return pi

    # mujoco
    if environ == 'mujoco':
        from utils.cmd import make_mujoco_env

        sess = Model().init_session(num_cpu=1).__enter__()

        def policy_fn(name, ob_space, ac_space):
            return PPO1Mlp(
                name=name,
                ob_space=ob_space,
                ac_space=ac_space,
                hid_size=64, num_hid_layers=2
            )
        env = make_mujoco_env(env_id, seed)
        pi = PPOSGD(
            env,
            policy_fn,
            env.observation_space,
            env.action_space,
            max_timesteps=num_timesteps,
            timesteps_per_actorbatch=2048,
            clip_param=0.2,
            entcoeff=0.0,
            optim_epochs=10,
            optim_stepsize=3e-4,
            optim_batchsize=64,
            gamma=0.99, lam=0.95,
            schedule='linear',
            )
        env.close()
        sess.close()
        return pi

    if environ == 'humanoid':
        import gym
        from utils.cmd import make_mujoco_env

        env_id = 'Humanoid-v2'

        class RewScale(gym.RewardWrapper):
            def __init__(self, env, scale):
                gym.RewardWrapper.__init__(self, env)
                self.scale = scale

            def reward(self, r):
                return r * self.scale

        sess = Model().init_session(num_cpu=1).__enter__()

        def policy_fn(name, ob_space, ac_space):
            return PPO1Mlp(
                name=name,
                ob_space=ob_space,
                ac_space=ac_space,
                hid_size=64,
                num_hid_layers=2
            )
        env = make_mujoco_env(env_id, seed)

        # parameters below were the best found in a simple random
        # search these are good enough to make humanoid walk, but
        # whether those are an absolute best or not is not certain
        env = RewScale(env, 0.1)
        pi = PPOSGD(
            env,
            policy_fn,
            env.observation_space,
            env.action_space,
            max_timesteps=num_timesteps,
            timesteps_per_actorbatch=2048,
            clip_param=0.2, entcoeff=0.0,
            optim_epochs=10,
            optim_stepsize=3e-4,
            optim_batchsize=64,
            gamma=0.99,
            lam=0.95,
            schedule='linear',
            )
        env.close()
        if model_path:
            Model().save_state(model_path)

        sess.close()
        return pi

    if environ == 'robotics':
        import mujoco_py
        from utils.cmd import make_robotics_env
        rank = MPI.COMM_WORLD.Get_rank()
        sess = Model().single_threaded_session()
        sess.__enter__()
        mujoco_py.ignore_mujoco_warnings().__enter__()
        workerseed = seed + 10000 * rank
        set_global_seeds(workerseed)
        env = make_robotics_env(env_id, workerseed, rank=rank)

        def policy_fn(name, ob_space, ac_space):
            return PPO1Mlp(
                name=name,
                ob_space=ob_space,
                ac_space=ac_space,
                hid_size=256, num_hid_layers=3
            )

        pi = PPOSGD(
            env,
            policy_fn,
            env.observation_space,
            env.action_space,
            max_timesteps=num_timesteps,
            timesteps_per_actorbatch=2048,
            clip_param=0.2,
            entcoeff=0.0,
            optim_epochs=5,
            optim_stepsize=3e-4,
            optim_batchsize=256,
            gamma=0.99,
            lam=0.95,
            schedule='linear',
            )
        env.close()
        sess.close()
        return pi


def main():
    parser = arg_parser()
    parser.add_argument('--platform', help='environment choice',
                        choices=['atari', 'mujoco', 'humanoid', 'robotics'],
                        default='atari')
    platform_args, environ_args = parser.parse_known_args()
    platform = platform_args.platform

    # atari
    if platform == 'atari':
        args = atari_arg_parser().parse_known_args()[0]
        pi = fit(
            platform,
            args.env,
            num_timesteps=args.num_timesteps,
            seed=args.seed
        )

    # mujoco
    if platform == 'mujoco':
        args = mujoco_arg_parser().parse_known_args()[0]
        logger.configure()
        pi = fit(
            platform,
            args.env,
            num_timesteps=args.num_timesteps,
            seed=args.seed
        )

    # robotics
    if platform == 'robotics':
        args = robotics_arg_parser().parse_known_args()[0]
        pi = fit(
            platform,
            args.env,
            num_timesteps=args.num_timesteps,
            seed=args.seed
        )

    # humanoids
    if platform == 'humanoid':
        logger.configure()
        parser = mujoco_arg_parser()
        parser.add_argument(
            '--model-path',
            default=os.path.join(logger.get_dir(), 'humanoid_policy')
        )
        parser.set_defaults(num_timesteps=int(2e7))

        args = parser.parse_known_args()[0]

        if not args.play:
            # train the model
            pi = fit(
                platform,
                args.env,
                num_timesteps=args.num_timesteps,
                seed=args.seed,
                model_path=args.model_path
            )
        else:
            # construct the model object, load pre-trained model and render
            from utils.cmd import make_mujoco_env
            pi = fit(
                platform,
                args.evn,
                num_timesteps=1,
                seed=args.seed
            )
            Model().load_state(args.model_path)
            env = make_mujoco_env('Humanoid-v2', seed=0)

            ob = env.reset()
            while True:
                action = pi.act(stochastic=False, ob=ob)[0]
                ob, _, done, _ = env.step(action)
                env.render()
                if done:
                    ob = env.reset()


if __name__ == '__main__':
    main()
