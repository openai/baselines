"""
Disclaimer: this code is highly based on trpo_mpi at @openai/stable_baselines and @openai/imitation
"""

import argparse
import os
import logging

from mpi4py import MPI
from tqdm import tqdm
import numpy as np
import gym

from stable_baselines.gail import mlp_policy, behavior_clone
from stable_baselines.trpo_mpi.trpo_mpi import TRPO
from stable_baselines.common import set_global_seeds, tf_util
from stable_baselines.common.misc_util import boolean_flag
from stable_baselines import bench, logger
from stable_baselines.gail.dataset.mujocodset import MujocoDset
from stable_baselines.gail.adversary import TransitionClassifier


def argsparser():
    """
    get an argument parser for training mujoco on gail

    :return: (ArgumentParser)
    """
    parser = argparse.ArgumentParser("Tensorflow Implementation of GAIL")
    parser.add_argument('--env_id', help='environment ID', default='Hopper-v2')
    parser.add_argument('--seed', help='RNG seed', type=int, default=0)
    parser.add_argument('--expert_path', type=str, default='data/deterministic.trpo.Hopper.0.00.npz')
    parser.add_argument('--checkpoint_dir', help='the directory to save model', default='checkpoint')
    parser.add_argument('--log_dir', help='the directory to save log file', default='log')
    parser.add_argument('--load_model_path', help='if provided, load the model', type=str, default=None)
    # Task
    parser.add_argument('--task', type=str, choices=['train', 'evaluate', 'sample'], default='train')
    # for evaluatation
    boolean_flag(parser, 'stochastic_policy', default=False, help_msg='use stochastic/deterministic policy to evaluate')
    boolean_flag(parser, 'save_sample', default=False, help_msg='save the trajectories or not')
    #  Mujoco Dataset Configuration
    parser.add_argument('--traj_limitation', type=int, default=-1)
    # Optimization Configuration
    parser.add_argument('--g_step', help='number of steps to train policy in each epoch', type=int, default=3)
    parser.add_argument('--d_step', help='number of steps to train discriminator in each epoch', type=int, default=1)
    # Network Configuration (Using MLP Policy)
    parser.add_argument('--policy_hidden_size', type=int, default=100)
    parser.add_argument('--adversary_hidden_size', type=int, default=100)
    # Algorithms Configuration
    parser.add_argument('--algo', type=str, choices=['trpo', 'ppo'], default='trpo')
    parser.add_argument('--max_kl', type=float, default=0.01)
    parser.add_argument('--policy_entcoeff', help='entropy coefficiency of policy', type=float, default=0)
    parser.add_argument('--adversary_entcoeff', help='entropy coefficiency of discriminator', type=float, default=1e-3)
    # Traing Configuration
    parser.add_argument('--save_per_iter', help='save model every xx iterations', type=int, default=100)
    parser.add_argument('--num_timesteps', help='number of timesteps per episode', type=int, default=5e6)
    # Behavior Cloning
    boolean_flag(parser, 'pretrained', default=False, help_msg='Use BC to pretrain')
    parser.add_argument('--bc_max_iter', help='Max iteration for training BC', type=int, default=1e4)
    return parser.parse_args()


def get_task_name(args):
    """
    get the task name

    :param args: (ArgumentParser) the training argument
    :return: (str) the task name
    """
    task_name = args.algo + "_gail."
    if args.pretrained:
        task_name += "with_pretrained."
    if args.traj_limitation != np.inf:
        task_name += "transition_limitation_%d." % args.traj_limitation
    task_name += args.env_id.split("-")[0]
    task_name = task_name + ".g_step_" + str(args.g_step) + ".d_step_" + str(args.d_step) + \
        ".policy_entcoeff_" + str(args.policy_entcoeff) + ".adversary_entcoeff_" + str(args.adversary_entcoeff)
    task_name += ".seed_" + str(args.seed)
    return task_name


def main(args):
    """
    start training the model

    :param args: (ArgumentParser) the training argument
    """
    with tf_util.make_session(num_cpu=1):
        set_global_seeds(args.seed)
        env = gym.make(args.env_id)

        def policy_fn(name, ob_space, ac_space, reuse=False, placeholders=None, sess=None):
            return mlp_policy.MlpPolicy(name=name, ob_space=ob_space, ac_space=ac_space, reuse=reuse, sess=sess,
                                        hid_size=args.policy_hidden_size, num_hid_layers=2, placeholders=placeholders)
        env = bench.Monitor(env, logger.get_dir() and
                            os.path.join(logger.get_dir(), "monitor.json"))
        env.seed(args.seed)
        gym.logger.setLevel(logging.WARN)
        task_name = get_task_name(args)
        args.checkpoint_dir = os.path.join(args.checkpoint_dir, task_name)
        args.log_dir = os.path.join(args.log_dir, task_name)

        if args.task == 'train':
            dataset = MujocoDset(expert_path=args.expert_path, traj_limitation=args.traj_limitation)
            reward_giver = TransitionClassifier(env, args.adversary_hidden_size, entcoeff=args.adversary_entcoeff)
            train(env, args.seed, policy_fn, reward_giver, dataset, args.algo, args.g_step, args.d_step,
                  args.policy_entcoeff, args.num_timesteps, args.save_per_iter, args.checkpoint_dir, args.pretrained,
                  args.bc_max_iter, task_name)
        elif args.task == 'evaluate':
            runner(env,
                   policy_fn,
                   args.load_model_path,
                   timesteps_per_batch=1024,
                   number_trajs=10,
                   stochastic_policy=args.stochastic_policy,
                   save=args.save_sample
                   )
        else:
            raise NotImplementedError
        env.close()


def train(env, seed, policy_fn, reward_giver, dataset, algo, g_step, d_step, policy_entcoeff, num_timesteps,
          save_per_iter, checkpoint_dir, pretrained, bc_max_iter, task_name=None):
    """
    train gail on mujoco

    :param env: (Gym Environment) the environment
    :param seed: (int) the initial random seed
    :param policy_fn: (function (str, Gym Space, Gym Space, bool): MLPPolicy) policy generator
    :param reward_giver: (TransitionClassifier) the reward predicter from obsevation and action
    :param dataset: (MujocoDset) the dataset manager
    :param algo: (str) the algorithm type (only 'trpo' is supported)
    :param g_step: (int) number of steps to train policy in each epoch
    :param d_step: (int) number of steps to train discriminator in each epoch
    :param policy_entcoeff: (float) the weight of the entropy loss for the policy
    :param num_timesteps: (int) the number of timesteps to run
    :param save_per_iter: (int) the number of iterations before saving
    :param checkpoint_dir: (str) the location for saving checkpoints
    :param pretrained: (bool) use a pretrained behavior clone
    :param bc_max_iter: (int) the maximum number of training iterations for the behavior clone
    :param task_name: (str) the name of the task (can be None)
    """

    pretrained_weight = None
    if pretrained and (bc_max_iter > 0):
        # Pretrain with behavior cloning
        pretrained_weight = behavior_clone.learn(env, policy_fn, dataset, max_iters=bc_max_iter)

    if algo == 'trpo':
        # Set up for MPI seed
        rank = MPI.COMM_WORLD.Get_rank()
        if rank != 0:
            logger.set_level(logger.DISABLED)
        workerseed = seed + 10000 * MPI.COMM_WORLD.Get_rank()
        set_global_seeds(workerseed)
        env.seed(workerseed)
        model = TRPO(policy_fn, env, timesteps_per_batch=1024, max_kl=0.01, cg_iters=10, gamma=0.995, lam=0.97,
                     entcoeff=policy_entcoeff, cg_damping=0.1, vf_stepsize=1e-3, vf_iters=5, _init_setup_model=False)

        # GAIL param
        model.pretrained_weight = pretrained_weight
        model.reward_giver = reward_giver
        model.expert_dataset = dataset
        model.save_per_iter = save_per_iter
        model.checkpoint_dir = checkpoint_dir
        model.g_step = g_step
        model.d_step = d_step
        model.task_name = task_name
        model.using_gail = True
        model.setup_model()

        model.learn(total_timesteps=num_timesteps)
    else:
        raise NotImplementedError


def runner(env, policy_func, load_model_path, timesteps_per_batch, number_trajs,
           stochastic_policy, save=False, reuse=False):
    """
    run the training for all the trajectories

    :param env: (Gym Environment) the environment
    :param policy_func: (function (str, Gym Space, Gym Space, bool): MLPPolicy) policy generator
    :param load_model_path: (str) the path to the model
    :param timesteps_per_batch: (int) the number of timesteps to run per batch (horizon)
    :param number_trajs: (int) the number of trajectories to run
    :param stochastic_policy: (bool) use a stochastic policy
    :param save: (bool) save the policy
    :param reuse: (bool) allow reuse of the graph
    :return: (float, float) average trajectory lenght, average trajectory reward
    """

    # Setup network
    # ----------------------------------------
    ob_space = env.observation_space
    ac_space = env.action_space
    policy = policy_func("pi", ob_space, ac_space, reuse=reuse)
    tf_util.initialize()
    # Prepare for rollouts
    # ----------------------------------------
    tf_util.load_state(load_model_path)

    obs_list = []
    acs_list = []
    len_list = []
    ret_list = []
    for _ in tqdm(range(number_trajs)):
        traj = traj_1_generator(policy, env, timesteps_per_batch, stochastic=stochastic_policy)
        obs, acs, ep_len, ep_ret = traj['ob'], traj['ac'], traj['ep_len'], traj['ep_ret']
        obs_list.append(obs)
        acs_list.append(acs)
        len_list.append(ep_len)
        ret_list.append(ep_ret)
    if stochastic_policy:
        print('stochastic policy:')
    else:
        print('deterministic policy:')
    if save:
        filename = load_model_path.split('/')[-1] + '.' + env.spec.id
        np.savez(filename, obs=np.array(obs_list), acs=np.array(acs_list),
                 lens=np.array(len_list), rets=np.array(ret_list))
    avg_len = sum(len_list)/len(len_list)
    avg_ret = sum(ret_list)/len(ret_list)
    print("Average length:", avg_len)
    print("Average return:", avg_ret)
    return avg_len, avg_ret


def traj_1_generator(policy, env, horizon, stochastic):
    """
    Sample one trajectory (until trajectory end)

    :param policy: (MLPPolicy) the policy
    :param env: (Gym Environment) the environment
    :param horizon: (int) the search horizon
    :param stochastic: (bool) use a stochastic policy
    :return: (dict) the trajectory
    """

    step = 0
    env.action_space.sample()  # not used, just so we have the datatype
    new = True  # marks if we're on first timestep of an episode

    observation = env.reset()
    cur_ep_ret = 0  # return in current episode
    cur_ep_len = 0  # len of current episode

    # Initialize history arrays
    observations = []
    rewards = []
    news = []
    actions = []

    while True:
        acttion, _ = policy.act(stochastic, observation)
        observations.append(observation)
        news.append(new)
        actions.append(acttion)

        observation, reward, new, _ = env.step(acttion)
        rewards.append(reward)

        cur_ep_ret += reward
        cur_ep_len += 1
        if new or step >= horizon:
            break
        step += 1

    observations = np.array(observations)
    rewards = np.array(rewards)
    news = np.array(news)
    actions = np.array(actions)
    traj = {"ob": observations, "rew": rewards, "new": news, "ac": actions,
            "ep_ret": cur_ep_ret, "ep_len": cur_ep_len}
    return traj


if __name__ == '__main__':
    args = argsparser()
    main(args)
