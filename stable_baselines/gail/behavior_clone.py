"""
The code is used to train BC imitator, or pretrained GAIL imitator
"""
import os
import argparse
import tempfile
import logging

from tqdm import tqdm
import gym
import tensorflow as tf

from stable_baselines.gail import mlp_policy
from stable_baselines import logger, bench
from stable_baselines.common import set_global_seeds, tf_util
from stable_baselines.common.misc_util import boolean_flag
from stable_baselines.common.mpi_adam import MpiAdam
from stable_baselines.gail.run_mujoco import runner
from stable_baselines.gail.dataset.mujocodset import MujocoDset


def argsparser():
    """
    make a behavior cloning argument parser

    :return: (ArgumentParser)
    """
    parser = argparse.ArgumentParser("Tensorflow Implementation of Behavior Cloning")
    parser.add_argument('--env_id', help='environment ID', default='Hopper-v1')
    parser.add_argument('--seed', help='RNG seed', type=int, default=0)
    parser.add_argument('--expert_path', type=str, default='data/deterministic.trpo.Hopper.0.00.npz')
    parser.add_argument('--checkpoint_dir', help='the directory to save model', default='checkpoint')
    parser.add_argument('--log_dir', help='the directory to save log file', default='log')
    #  Mujoco Dataset Configuration
    parser.add_argument('--traj_limitation', type=int, default=-1)
    # Network Configuration (Using MLP Policy)
    parser.add_argument('--policy_hidden_size', type=int, default=100)
    # for evaluatation
    boolean_flag(parser, 'stochastic_policy', default=False, help_msg='use stochastic/deterministic policy to evaluate')
    boolean_flag(parser, 'save_sample', default=False, help_msg='save the trajectories or not')
    parser.add_argument('--BC_max_iter', help='Max iteration for training BC', type=int, default=1e5)
    return parser.parse_args()


def learn(env, policy_func, dataset, optim_batch_size=128, max_iters=1e4, adam_epsilon=1e-5, optim_stepsize=3e-4,
          ckpt_dir=None, task_name=None, verbose=False):
    """
    Learn a behavior clone policy, and return the save location

    :param env: (Gym Environment) the environment
    :param policy_func: (function (str, Gym Space, Gym Space): TensorFlow Tensor) creates the policy
    :param dataset: (Dset or MujocoDset) the dataset manager
    :param optim_batch_size: (int) the batch size
    :param max_iters: (int) the maximum number of iterations
    :param adam_epsilon: (float) the epsilon value for the adam optimizer
    :param optim_stepsize: (float) the optimizer stepsize
    :param ckpt_dir: (str) the save directory, can be None for temporary directory
    :param task_name: (str) the save name, can be None for saving directly to the directory name
    :param verbose: (bool)
    :return: (str) the save location for the TensorFlow model
    """

    val_per_iter = int(max_iters/10)
    ob_space = env.observation_space
    ac_space = env.action_space
    policy = policy_func("pi", ob_space, ac_space)  # Construct network for new policy
    # placeholder
    obs_ph = policy.obs_ph
    action_ph = policy.pdtype.sample_placeholder([None])
    stochastic_ph = policy.stochastic_ph
    loss = tf.reduce_mean(tf.square(action_ph - policy.ac))
    var_list = policy.get_trainable_variables()
    adam = MpiAdam(var_list, epsilon=adam_epsilon)
    lossandgrad = tf_util.function([obs_ph, action_ph, stochastic_ph], [loss] + [tf_util.flatgrad(loss, var_list)])

    tf_util.initialize()
    adam.sync()
    logger.log("Pretraining with Behavior Cloning...")
    for iter_so_far in tqdm(range(int(max_iters))):
        ob_expert, ac_expert = dataset.get_next_batch(optim_batch_size, 'train')
        train_loss, grad = lossandgrad(ob_expert, ac_expert, True)
        adam.update(grad, optim_stepsize)
        if verbose and iter_so_far % val_per_iter == 0:
            ob_expert, ac_expert = dataset.get_next_batch(-1, 'val')
            val_loss, _ = lossandgrad(ob_expert, ac_expert, True)
            logger.log("Training loss: {}, Validation loss: {}".format(train_loss, val_loss))

    if ckpt_dir is None:
        savedir_fname = tempfile.TemporaryDirectory().name
    else:
        savedir_fname = os.path.join(ckpt_dir, task_name)
    tf_util.save_state(savedir_fname, var_list=policy.get_variables())
    return savedir_fname


def get_task_name(args):
    """
    Get the task name

    :param args: (ArgumentParser) the training argument
    :return: (str) the task name
    """
    task_name = 'BC'
    task_name += '.{}'.format(args.env_id.split("-")[0])
    task_name += '.traj_limitation_{}'.format(args.traj_limitation)
    task_name += ".seed_{}".format(args.seed)
    return task_name


def main(args):
    """
    start training the model

    :param args: (ArgumentParser) the training argument
    """
    with tf_util.make_session(num_cpu=1):
        set_global_seeds(args.seed)
        env = gym.make(args.env_id)

        def policy_fn(name, ob_space, ac_space, reuse=False, sess=None):
            return mlp_policy.MlpPolicy(name=name, ob_space=ob_space, ac_space=ac_space, sess=sess,
                                        reuse=reuse, hid_size=args.policy_hidden_size, num_hid_layers=2)
        env = bench.Monitor(env, logger.get_dir() and
                            os.path.join(logger.get_dir(), "monitor.json"))
        env.seed(args.seed)
        gym.logger.setLevel(logging.WARN)
        task_name = get_task_name(args)
        args.checkpoint_dir = os.path.join(args.checkpoint_dir, task_name)
        args.log_dir = os.path.join(args.log_dir, task_name)
        dataset = MujocoDset(expert_path=args.expert_path, traj_limitation=args.traj_limitation)
        savedir_fname = learn(env, policy_fn, dataset, max_iters=args.BC_max_iter, ckpt_dir=args.checkpoint_dir,
                              task_name=task_name, verbose=True)
        runner(env,
               policy_fn,
               savedir_fname,
               timesteps_per_batch=1024,
               number_trajs=10,
               stochastic_policy=args.stochastic_policy,
               save=args.save_sample,
               reuse=True)


if __name__ == '__main__':
    args = argsparser()
    main(args)
