import os
import sys
from subprocess import CalledProcessError

import click
import numpy as np
import json
from mpi4py import MPI

from stable_baselines import logger
from stable_baselines.common import set_global_seeds, tf_util
from stable_baselines.common.mpi_moments import mpi_moments
import stable_baselines.her.experiment.config as config
from stable_baselines.her.rollout import RolloutWorker
from stable_baselines.her.util import mpi_fork


def mpi_average(value):
    """
    calculate the average from the array, using MPI

    :param value: (np.ndarray) the array
    :return: (float) the average
    """
    if len(value) == 0:
        value = [0.]
    if not isinstance(value, list):
        value = [value]
    return mpi_moments(np.array(value))[0]


def train(policy, rollout_worker, evaluator, n_epochs, n_test_rollouts, n_cycles, n_batches, policy_save_interval,
          save_policies):
    """
    train the given policy

    :param policy: (her.DDPG) the policy to train
    :param rollout_worker: (RolloutWorker) Rollout worker generates experience for training.
    :param evaluator: (RolloutWorker)  Rollout worker for evalutation
    :param n_epochs: (int) the number of epochs
    :param n_test_rollouts: (int) the number of for the evalutation RolloutWorker
    :param n_cycles: (int) the number of cycles for training per epoch
    :param n_batches: (int) the batch size
    :param policy_save_interval: (int) the interval with which policy pickles are saved.
        If set to 0, only the best and latest policy will be pickled.
    :param save_policies: (bool) whether or not to save the policies
    """
    rank = MPI.COMM_WORLD.Get_rank()

    latest_policy_path = os.path.join(logger.get_dir(), 'policy_latest.pkl')
    best_policy_path = os.path.join(logger.get_dir(), 'policy_best.pkl')
    periodic_policy_path = os.path.join(logger.get_dir(), 'policy_{}.pkl')

    logger.info("Training...")
    best_success_rate = -1
    for epoch in range(n_epochs):
        # train
        rollout_worker.clear_history()
        for _ in range(n_cycles):
            episode = rollout_worker.generate_rollouts()
            policy.store_episode(episode)
            for _ in range(n_batches):
                policy.train_step()
            policy.update_target_net()

        # test
        evaluator.clear_history()
        for _ in range(n_test_rollouts):
            evaluator.generate_rollouts()

        # record logs
        logger.record_tabular('epoch', epoch)
        for key, val in evaluator.logs('test'):
            logger.record_tabular(key, mpi_average(val))
        for key, val in rollout_worker.logs('train'):
            logger.record_tabular(key, mpi_average(val))
        for key, val in policy.logs():
            logger.record_tabular(key, mpi_average(val))

        if rank == 0:
            logger.dump_tabular()

        # save the policy if it's better than the previous ones
        success_rate = mpi_average(evaluator.current_success_rate())
        if rank == 0 and success_rate >= best_success_rate and save_policies:
            best_success_rate = success_rate
            logger.info('New best success rate: {}. Saving policy to {} ...'
                        .format(best_success_rate, best_policy_path))
            evaluator.save_policy(best_policy_path)
            evaluator.save_policy(latest_policy_path)
        if rank == 0 and policy_save_interval > 0 and epoch % policy_save_interval == 0 and save_policies:
            policy_path = periodic_policy_path.format(epoch)
            logger.info('Saving periodic policy to {} ...'.format(policy_path))
            evaluator.save_policy(policy_path)

        # make sure that different threads have different seeds
        local_uniform = np.random.uniform(size=(1,))
        root_uniform = local_uniform.copy()
        MPI.COMM_WORLD.Bcast(root_uniform, root=0)
        if rank != 0:
            assert local_uniform[0] != root_uniform[0]


def launch(env, logdir, n_epochs, num_cpu, seed, replay_strategy, policy_save_interval, clip_return,
           override_params=None, save_policies=True):
    """
    launch training with mpi

    :param env: (str) environment ID
    :param logdir: (str) the log directory
    :param n_epochs: (int) the number of training epochs
    :param num_cpu: (int) the number of CPUs to run on
    :param seed: (int) the initial random seed
    :param replay_strategy: (str) the type of replay strategy ('future' or 'none')
    :param policy_save_interval: (int) the interval with which policy pickles are saved.
        If set to 0, only the best and latest policy will be pickled.
    :param clip_return: (float): clip returns to be in [-clip_return, clip_return]
    :param override_params: (dict) override any parameter for training
    :param save_policies: (bool) whether or not to save the policies
    """

    if override_params is None:
        override_params = {}
    # Fork for multi-CPU MPI implementation.
    if num_cpu > 1:
        try:
            whoami = mpi_fork(num_cpu, ['--bind-to', 'core'])
        except CalledProcessError:
            # fancy version of mpi call failed, try simple version
            whoami = mpi_fork(num_cpu)

        if whoami == 'parent':
            sys.exit(0)
        tf_util.single_threaded_session().__enter__()
    rank = MPI.COMM_WORLD.Get_rank()

    # Configure logging
    if rank == 0:
        if logdir or logger.get_dir() is None:
            logger.configure(folder=logdir)
    else:
        logger.configure()
    logdir = logger.get_dir()
    assert logdir is not None
    os.makedirs(logdir, exist_ok=True)

    # Seed everything.
    rank_seed = seed + 1000000 * rank
    set_global_seeds(rank_seed)

    # Prepare params.
    params = config.DEFAULT_PARAMS
    params['env_name'] = env
    params['replay_strategy'] = replay_strategy
    if env in config.DEFAULT_ENV_PARAMS:
        params.update(config.DEFAULT_ENV_PARAMS[env])  # merge env-specific parameters in
    params.update(**override_params)  # makes it possible to override any parameter
    with open(os.path.join(logger.get_dir(), 'params.json'), 'w') as file_handler:
        json.dump(params, file_handler)
    params = config.prepare_params(params)
    config.log_params(params, logger_input=logger)

    if num_cpu == 1:
        logger.warn()
        logger.warn('*** Warning ***')
        logger.warn(
            'You are running HER with just a single MPI worker. This will work, but the ' +
            'experiments that we report in Plappert et al. (2018, https://arxiv.org/abs/1802.09464) ' +
            'were obtained with --num_cpu 19. This makes a significant difference and if you ' +
            'are looking to reproduce those results, be aware of this. Please also refer to ' +
            'https://github.com/openai/stable_baselines/issues/314 for further details.')
        logger.warn('****************')
        logger.warn()

    dims = config.configure_dims(params)
    policy = config.configure_ddpg(dims=dims, params=params, clip_return=clip_return)

    rollout_params = {
        'exploit': False,
        'use_target_net': False,
        # 'use_demo_states': True,
        'compute_q': False,
        'time_horizon': params['time_horizon'],
    }

    eval_params = {
        'exploit': True,
        'use_target_net': params['test_with_polyak'],
        # 'use_demo_states': False,
        'compute_q': True,
        'time_horizon': params['time_horizon'],
    }

    for name in ['time_horizon', 'rollout_batch_size', 'gamma', 'noise_eps', 'random_eps']:
        rollout_params[name] = params[name]
        eval_params[name] = params[name]

    rollout_worker = RolloutWorker(params['make_env'], policy, dims, logger, **rollout_params)
    rollout_worker.seed(rank_seed)

    evaluator = RolloutWorker(params['make_env'], policy, dims, logger, **eval_params)
    evaluator.seed(rank_seed)

    train(
        policy=policy, rollout_worker=rollout_worker,
        evaluator=evaluator, n_epochs=n_epochs, n_test_rollouts=params['n_test_rollouts'],
        n_cycles=params['n_cycles'], n_batches=params['n_batches'],
        policy_save_interval=policy_save_interval, save_policies=save_policies)


@click.command()
@click.option('--env', type=str, default='FetchReach-v1',
              help='the name of the OpenAI Gym environment that you want to train on')
@click.option('--logdir', type=str, default=None,
              help='the path to where logs and policy pickles should go. If not specified, creates a folder in /tmp/')
@click.option('--n_epochs', type=int, default=50, help='the number of training epochs to run')
@click.option('--num_cpu', type=int, default=1, help='the number of CPU cores to use (using MPI)')
@click.option('--seed', type=int, default=0,
              help='the random seed used to seed both the environment and the training code')
@click.option('--policy_save_interval', type=int, default=5,
              help='the interval with which policy pickles are saved. '
                   'If set to 0, only the best and latest policy will be pickled.')
@click.option('--replay_strategy', type=click.Choice(['future', 'none']), default='future',
              help='the HER replay strategy to be used. "future" uses HER, "none" disables HER.')
@click.option('--clip_return', type=int, default=1, help='whether or not returns should be clipped')
def main(**kwargs):
    """
    run launch for MPI HER DDPG training

    :param kwargs: (dict) the launch kwargs
    """
    launch(**kwargs)


if __name__ == '__main__':
    main()
