from gym.envs.registration import register


def _merge(a, b):
    a.update(b)
    return a

register(
    id='GraspBlock-v0',
    entry_point='gym_grasp.envs:GraspBlockEnv',
    max_episode_steps=100,
)