from baselines.deepq import models  # noqa F401
from baselines.deepq.deepq_learner import DEEPQ  # noqa F401
from baselines.deepq.deepq import learn  # noqa F401
from baselines.deepq.replay_buffer import ReplayBuffer, PrioritizedReplayBuffer  # noqa F401

def wrap_atari_dqn(env):
    from baselines.common.atari_wrappers import wrap_deepmind
    return wrap_deepmind(env, frame_stack=True, scale=False)
