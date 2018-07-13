from baselines.gail.trpo_mpi import learn as base_learn


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
    return base_learn(env, policy_fn, timesteps_per_batch=timesteps_per_batch, max_kl=max_kl,
                      cg_iters=cg_iters, gamma=gamma, lam=lam, entcoeff=entcoeff, cg_damping=cg_damping,
                      vf_stepsize=vf_stepsize, vf_iters=vf_iters, max_timesteps=max_timesteps, max_episodes=max_episodes,
                      max_iters=max_iters, callback=callback, using_gail=False)
