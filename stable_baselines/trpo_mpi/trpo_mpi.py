from stable_baselines.gail.trpo_mpi import learn as base_learn


def learn(env, policy_fn, *,
          timesteps_per_batch,  # what to train on
          max_kl, cg_iters,
          gamma, lam,  # advantage estimation
          entcoeff=0.0,
          cg_damping=1e-2,
          vf_stepsize=3e-4,
          vf_iters=3,
          max_timesteps=0, max_episodes=0, max_iters=0,  # time constraint
          callback=None):
    """
    learns a TRPO policy using the given environment

    :param env: (Gym Environment) the environment
    :param policy_fn: (function (str, Gym Space, Gym Space, bool): MLPPolicy) policy generator
    :param timesteps_per_batch: (int) the number of timesteps to run per batch (horizon)
    :param max_kl: (float) the kullback leiber loss threashold
    :param cg_iters: (int) the number of iterations for the conjugate gradient calculation
    :param gamma: (float) the discount value
    :param lam: (float) GAE factor
    :param entcoeff: (float) the weight for the entropy loss
    :param cg_damping: (float) the compute gradient dampening factor
    :param vf_stepsize: (float) the value function stepsize
    :param vf_iters: (int) the value function's number iterations for learning
    :param max_timesteps: (int) the maximum number of timesteps before halting
    :param max_episodes: (int) the maximum number of episodes before halting
    :param max_iters: (int) the maximum number of training iterations  before halting
    :param callback: (function (dict, dict)) the call back function, takes the local and global attribute dictionary
    """
    base_learn(env, policy_fn, timesteps_per_batch=timesteps_per_batch, max_kl=max_kl, cg_iters=cg_iters, gamma=gamma,
               lam=lam, entcoeff=entcoeff, cg_damping=cg_damping, vf_stepsize=vf_stepsize, vf_iters=vf_iters,
               max_timesteps=max_timesteps, max_episodes=max_episodes, max_iters=max_iters, callback=callback,
               using_gail=False)
