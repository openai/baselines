#!/usr/bin/env python3
import os

import gym

from stable_baselines.ppo1 import PPO1
from stable_baselines.common.cmd_util import make_mujoco_env, mujoco_arg_parser
from stable_baselines.common import tf_util
from stable_baselines.common.policies import MlpPolicy
from stable_baselines import logger


def train(num_timesteps, seed, model_path=None):
    """
    Train PPO1 model for the Humanoid environment, for testing purposes

    :param num_timesteps: (int) The total number of samples
    :param seed: (int) The initial seed for training
    :param model_path: (str) path to the model
    """
    env_id = 'Humanoid-v2'

    env = make_mujoco_env(env_id, seed)

    # parameters below were the best found in a simple random search
    # these are good enough to make humanoid walk, but whether those are
    # an absolute best or not is not certain
    env = RewScale(env, 0.1)
    model = PPO1(MlpPolicy, env, timesteps_per_actorbatch=2048, clip_param=0.2, entcoeff=0.0, optim_epochs=10,
                 optim_stepsize=3e-4, optim_batchsize=64, gamma=0.99, lam=0.95, schedule='linear')
    model.learn(total_timesteps=num_timesteps)
    env.close()
    if model_path:
        tf_util.save_state(model_path)

    return model


class RewScale(gym.RewardWrapper):
    def __init__(self, env, scale):
        gym.RewardWrapper.__init__(self, env)
        self.scale = scale

    def reward(self, _reward):
        return _reward * self.scale


def main():
    """
    Runs the test
    """
    logger.configure()
    parser = mujoco_arg_parser()
    parser.add_argument('--model-path', default=os.path.join(logger.get_dir(), 'humanoid_policy'))
    parser.set_defaults(num_timesteps=int(2e7))

    args = parser.parse_args()

    if not args.play:
        # train the model
        train(num_timesteps=args.num_timesteps, seed=args.seed, model_path=args.model_path)
    else:
        # construct the model object, load pre-trained model and render
        model = train(num_timesteps=1, seed=args.seed)
        tf_util.load_state(args.model_path)
        env = make_mujoco_env('Humanoid-v2', seed=0)

        obs = env.reset()
        while True:
            action = model.policy.act(stochastic=False, obs=obs)[0]
            obs, _, done, _ = env.step(action)
            env.render()
            if done:
                obs = env.reset()


if __name__ == '__main__':
    main()
