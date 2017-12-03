import tempfile
from tqdm import tqdm

import tensorflow as tf

import baselines.common.tf_util as U
from baselines import logger
from baselines.common.mpi_adam import MpiAdam


def evaluate(env, policy_func, load_model_path, stochastic_policy=False, number_trajs=10):
    from algo.trpo_mpi import traj_episode_generator
    ob_space = env.observation_space
    ac_space = env.action_space
    pi = policy_func("pi", ob_space, ac_space)  # Construct network for new policy
    # placeholder
    ep_gen = traj_episode_generator(pi, env, 1024, stochastic=stochastic_policy)
    U.load_state(load_model_path)
    len_list = []
    ret_list = []
    for _ in tqdm(range(number_trajs)):
        traj = ep_gen.__next__()
        ep_len, ep_ret = traj['ep_len'], traj['ep_ret']
        len_list.append(ep_len)
        ret_list.append(ep_ret)
    if stochastic_policy:
        print('stochastic policy:')
    else:
        print('deterministic policy:')
    print("Average length:", sum(len_list)/len(len_list))
    print("Average return:", sum(ret_list)/len(ret_list))


def learn(env, policy_func, dataset, optim_batch_size=128, max_iters=1e4,
          adam_epsilon=1e-5, optim_stepsize=3e-4,
          ckpt_dir=None, log_dir=None, task_name=None,
          verbose=False):

    val_per_iter = int(max_iters/10)
    ob_space = env.observation_space
    ac_space = env.action_space
    pi = policy_func("pi", ob_space, ac_space)  # Construct network for new policy
    # placeholder
    ob = U.get_placeholder_cached(name="ob")
    ac = pi.pdtype.sample_placeholder([None])
    stochastic = U.get_placeholder_cached(name="stochastic")
    loss = tf.reduce_mean(tf.square(ac-pi.ac))
    var_list = pi.get_trainable_variables()
    adam = MpiAdam(var_list, epsilon=adam_epsilon)
    lossandgrad = U.function([ob, ac, stochastic], [loss]+[U.flatgrad(loss, var_list)])

    U.initialize()
    adam.sync()
    logger.log("Pretraining with Behavior Cloning...")
    for iter_so_far in tqdm(range(int(max_iters))):
        ob_expert, ac_expert = dataset.get_next_batch(optim_batch_size, 'train')
        loss, g = lossandgrad(ob_expert, ac_expert, True)
        adam.update(g, optim_stepsize)
        if verbose and iter_so_far % val_per_iter == 0:
            ob_expert, ac_expert = dataset.get_next_batch(-1, 'val')
            loss, g = lossandgrad(ob_expert, ac_expert, False)
            logger.log("Validation:")
            logger.log("Loss: %f" % loss)

    savedir_fname = tempfile.TemporaryDirectory().name
    U.save_state(savedir_fname, var_list=pi.get_variables())
    return savedir_fname
