import time
from collections import deque
from contextlib import contextmanager

import tensorflow as tf
import numpy as np
from mpi4py import MPI

from baselines.common import explained_variance, zipsame, dataset, colorize
from baselines import logger
import baselines.common.tf_util as tf_util
from baselines.common.mpi_adam import MpiAdam
from baselines.common.cg import conjugate_gradient
from baselines.gail.trpo_mpi import traj_segment_generator, add_vtarg_and_adv, flatten_lists, learn as base_learn


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
