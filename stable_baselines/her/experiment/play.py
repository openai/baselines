import click
import pickle

import numpy as np

from stable_baselines import logger
from stable_baselines.common import set_global_seeds
import stable_baselines.her.experiment.config as config
from stable_baselines.her.rollout import RolloutWorker


@click.command()
@click.argument('policy_file', type=str)
@click.option('--seed', type=int, default=0)
@click.option('--n_test_rollouts', type=int, default=10)
@click.option('--render', type=int, default=1)
def main(policy_file, seed, n_test_rollouts, render):
    """
    run HER from a saved policy

    :param policy_file: (str) pickle path to a saved policy
    :param seed: (int) initial seed
    :param n_test_rollouts: (int) the number of test rollouts
    :param render: (bool) if rendering should be done
    """
    set_global_seeds(seed)

    # Load policy.
    with open(policy_file, 'rb') as file_handler:
        policy = pickle.load(file_handler)
    env_name = policy.info['env_name']

    # Prepare params.
    params = config.DEFAULT_PARAMS
    if env_name in config.DEFAULT_ENV_PARAMS:
        params.update(config.DEFAULT_ENV_PARAMS[env_name])  # merge env-specific parameters in
    params['env_name'] = env_name
    params = config.prepare_params(params)
    config.log_params(params, logger_input=logger)

    dims = config.configure_dims(params)

    eval_params = {
        'exploit': True,
        'use_target_net': params['test_with_polyak'],
        'compute_q': True,
        'rollout_batch_size': 1,
        'render': bool(render),
    }

    for name in ['time_horizon', 'gamma', 'noise_eps', 'random_eps']:
        eval_params[name] = params[name]

    evaluator = RolloutWorker(params['make_env'], policy, dims, logger, **eval_params)
    evaluator.seed(seed)

    # Run evaluation.
    evaluator.clear_history()
    for _ in range(n_test_rollouts):
        evaluator.generate_rollouts()

    # record logs
    for key, val in evaluator.logs('test'):
        logger.record_tabular(key, np.mean(val))
    logger.dump_tabular()


if __name__ == '__main__':
    main()
