from baselines.common.tests.envs.identity_env import DiscreteIdentityEnv


def test_discrete_nodelay():
    nsteps = 100
    eplen = 50
    env = DiscreteIdentityEnv(10, episode_len=eplen)
    ob = env.reset()
    for t in range(nsteps):
        action = env.action_space.sample()
        next_ob, rew, done, info = env.step(action)
        assert rew == (1 if action == ob else 0)
        if (t + 1) % eplen == 0:
            assert done
            next_ob = env.reset()
        else:
            assert not done
        ob = next_ob

def test_discrete_delay1():
    eplen = 50
    env = DiscreteIdentityEnv(10, episode_len=eplen, delay=1)
    ob = env.reset()
    prev_ob = None
    for t in range(eplen):
        action = env.action_space.sample()
        next_ob, rew, done, info = env.step(action)
        if t > 0:
            assert rew == (1 if action == prev_ob else 0)
        else:
            assert rew == 0
        prev_ob = ob
        ob = next_ob
        if t < eplen - 1:
            assert not done
    assert done
