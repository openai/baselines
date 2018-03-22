import os
import sys

import click
import numpy as np
import json
from mpi4py import MPI
import dill as pickle

from baselines import logger
from baselines.common import set_global_seeds
from baselines.common.mpi_moments import mpi_moments
import baselines.her.experiment.config as config
from baselines.her.rollout import RolloutWorker
from baselines.her.util import mpi_fork

def get_latest_pkl(logdir):
    if(logdir==None):
        print("enter with option --logdir")
    policy_list = os.listdir(logdir)
    latest_pkl_value = -1
    latest_pkl_file = None
    for pkl_file in policy_list:
        if(pkl_file.startswith("policy_")):
            pkl = pkl_file[7:]      #strip policy_
            pkl = pkl[:-4]      #strip .pkl
            if(pkl.isdigit()):
                pkl_value = int(pkl)
                if(pkl_value>latest_pkl_value):
                    latest_pkl_value = pkl_value       
                    latest_pkl_file = pkl_file
    return latest_pkl_file,latest_pkl_value

def mpi_average(value):
    if value == []:
        value = [0.]
    if not isinstance(value, list):
        value = [value]
    return mpi_moments(np.array(value))[0]


def train(policy, rollout_worker, evaluator,
          n_epochs, n_test_rollouts, n_cycles, n_batches, policy_save_interval,
          save_policies,retrainable=False,retrain=False,start_epoch=0, **kwargs):
    rank = MPI.COMM_WORLD.Get_rank()

    latest_policy_path = os.path.join(logger.get_dir(), 'policy_latest.pkl')
    best_policy_path = os.path.join(logger.get_dir(), 'policy_best.pkl')
    periodic_policy_path = os.path.join(logger.get_dir(), 'policy_{}.pkl')

    if(not(retrainable) or rank==0):
        logger.info("Training...")
    for epoch in range(start_epoch,n_epochs):
        # train
        rollout_worker.clear_history()
        for _ in range(n_cycles):
            episode = rollout_worker.generate_rollouts()
            policy.store_episode(episode)
            for _ in range(n_batches):
                policy.train()
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
            
        if(not(retrainable) or rank==0):
            logger.dump_tabular()

        # save the policy if it's better than the previous ones
        success_rate = mpi_average(evaluator.current_success_rate())
        if success_rate >= evaluator.policy.best_success_rate and save_policies:
            evaluator.policy.best_success_rate = success_rate
            if(not(retrainable) or rank==0):
                logger.info('New best success rate: {}. Saving policy to {} ...'.format(evaluator.policy.best_success_rate, best_policy_path))
            evaluator.policy.old_policy = True
            evaluator.save_policy(best_policy_path)
            evaluator.save_policy(latest_policy_path)
            evaluator.policy.old_policy = True

        if policy_save_interval > 0 and epoch % policy_save_interval == 0 and save_policies:
            policy_path = periodic_policy_path.format(epoch)
            if(not(retrainable) or rank==0):
                logger.info('Saving periodic policy to {} ...'.format(policy_path))
            evaluator.save_policy(policy_path)
            if(retrainable and epoch>0):
                old_policy_path = periodic_policy_path.format(epoch-policy_save_interval)
                print("Removing replay buffer from the previous pkl")
                with open(old_policy_path, 'rb') as f:
                    old_policy = pickle.load(f)
                old_policy.old_policy = True
                with open(old_policy_path, 'wb') as f:
                    pickle.dump(old_policy, f)  
                with open(policy_path, 'rb') as f:
                    evaluator.policy = pickle.load(f)
        # make sure that different threads have different seeds
        local_uniform = np.random.uniform(size=(1,))
        root_uniform = local_uniform.copy()
        MPI.COMM_WORLD.Bcast(root_uniform, root=0)
        if rank != 0:
            assert local_uniform[0] != root_uniform[0]


def launch(
    env_name, logdir, n_epochs, num_cpu, seed, replay_strategy, policy_save_interval, clip_return,batch_size,
    rollout_batch_size,retrainable,override_params={}, save_policies=True
):
    # Fork for multi-CPU MPI implementation.
    if num_cpu > 1:
        whoami = mpi_fork(num_cpu)
        if whoami == 'parent':
            sys.exit(0)
        import baselines.common.tf_util as U
        U.single_threaded_session().__enter__()
    rank = MPI.COMM_WORLD.Get_rank()
    
    # Configure logging
    if(retrainable):
        assert logdir is not None
        logdir=os.path.join(logdir,"cpu"+str(rank))
        os.makedirs(logdir, exist_ok=True)
        latest_policy_path , last_epoch = get_latest_pkl(logdir)
        retrain = False
        if(latest_policy_path is not None):
            retrain = True
        logger.configure(dir=logdir,retrain=retrain)
        logdir = logger.get_dir()
    else:
        if rank == 0:
            if logdir or logger.get_dir() is None:
                logger.configure(dir=logdir)
        else:
            logger.configure()
        logdir = logger.get_dir()
        assert logdir is not None
        os.makedirs(logdir, exist_ok=True)

    # Seed everything.
    rank_seed = seed + 1000000 * rank
    set_global_seeds(rank_seed)

    # Prepare params.
    if (retrainable and retrain):
        with open(os.path.join(logdir, 'params.json'), 'r') as f:
            params = json.load(f)
    else:   
        params = config.DEFAULT_PARAMS
        params['env_name'] = env_name
        params['replay_strategy'] = replay_strategy
        params['batch_size'] = batch_size
        params['rollout_batch_size'] = rollout_batch_size
        if env_name in config.DEFAULT_ENV_PARAMS:
            params.update(config.DEFAULT_ENV_PARAMS[env_name])  # merge env-specific parameters in
        params.update(**override_params)  # makes it possible to override any parameter
        with open(os.path.join(logger.get_dir(), 'params.json'), 'w') as f:
            json.dump(params, f)
    params = config.prepare_params(params)
    
    if(not(retrainable) or rank==0):
        config.log_params(params, logger=logger)

    if(not(retrainable) or not(retrain)):
        if(not(retrainable) or rank==0):
            if num_cpu == 1:
                logger.warn()
                logger.warn('*** Warning ***')
                logger.warn(
                    'You are running HER with just a single MPI worker. This will work, but the ' +
                    'experiments that we report in Plappert et al. (2018, https://arxiv.org/abs/1802.09464) ' +
                    'were obtained with --num_cpu 19. This makes a significant difference and if you ' +
                    'are looking to reproduce those results, be aware of this. Please also refer to ' + 
                    'https://github.com/openai/baselines/issues/314 for further details.')
                logger.warn('****************')
                logger.warn()

    dims = config.configure_dims(params)
    if(retrainable and retrain):
        with open(os.path.join(logdir,latest_policy_path), 'rb') as f:
            policy = pickle.load(f)
    else:
        policy = config.configure_ddpg(dims=dims, params=params, clip_return=clip_return)
    policy.retrainable = retrainable
    
    rollout_params = {
        'exploit': False,
        'use_target_net': False,
        'use_demo_states': True,
        'compute_Q': False,
        'T': params['T'],
    }

    eval_params = {
        'exploit': True,
        'use_target_net': params['test_with_polyak'],
        'use_demo_states': False,
        'compute_Q': True,
        'T': params['T'],
    }

    if(retrainable):
        rollout_params['n_episodes'] = (last_epoch*params['n_test_rollouts']*rollout_batch_size)
        eval_params['n_episodes'] = (last_epoch*params['n_test_rollouts']*rollout_batch_size)
        
    for name in ['T', 'rollout_batch_size', 'gamma', 'noise_eps', 'random_eps']:
        rollout_params[name] = params[name]
        eval_params[name] = params[name]

    rollout_worker = RolloutWorker(params['make_env'], policy, dims, logger, **rollout_params)
    rollout_worker.seed(rank_seed)

    evaluator = RolloutWorker(params['make_env'], policy, dims, logger, **eval_params)
    evaluator.seed(rank_seed)

    if(retrainable):
        train(
            logdir=logdir, policy=policy, rollout_worker=rollout_worker,
            evaluator=evaluator, n_epochs=n_epochs, n_test_rollouts=params['n_test_rollouts'],
            n_cycles=params['n_cycles'], n_batches=params['n_batches'],
            policy_save_interval=policy_save_interval, save_policies=save_policies,
            retrainable=True,retrain=retrain,start_epoch=last_epoch+1)
    else:    
        train(
            logdir=logdir, policy=policy, rollout_worker=rollout_worker,
            evaluator=evaluator, n_epochs=n_epochs, n_test_rollouts=params['n_test_rollouts'],
            n_cycles=params['n_cycles'], n_batches=params['n_batches'],
            policy_save_interval=policy_save_interval, save_policies=save_policies)

@click.command()
@click.option('--env_name', type=str, default='FetchReach-v0', help='the name of the OpenAI Gym environment that you want to train on')
@click.option('--logdir', type=str, default=None, help='the path to where logs and policy pickles should go. If not specified, creates a folder in /tmp/')
@click.option('--n_epochs', type=int, default=50, help='the number of training epochs to run')
@click.option('--num_cpu', type=int, default=1, help='the number of CPU cores to use (using MPI)')
@click.option('--seed', type=int, default=0, help='the random seed used to seed both the environment and the training code')
@click.option('--policy_save_interval', type=int, default=5, help='the interval with which policy pickles are saved. If set to 0, only the best and latest policy will be pickled.')
@click.option('--replay_strategy', type=click.Choice(['future', 'none']), default='future', help='the HER replay strategy to be used. "future" uses HER, "none" disables HER.')
@click.option('--clip_return', type=int, default=1, help='whether or not returns should be clipped')
@click.option('--batch_size', type=int, default=256, help='mini batch_size')
@click.option('--rollout_batch_size', type=int, default=2, help='number of rollouts generated')
@click.option('--retrainable',is_flag=True,help='whether or not to store replay buffer. If stored than can retrain later using this option again. Necessary to pass logdir with this option')

def main(**kwargs):
    launch(**kwargs)


if __name__ == '__main__':
    main()
