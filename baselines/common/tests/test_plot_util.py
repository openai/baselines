# smoke tests of plot_util
from baselines.common import plot_util as pu
from baselines.common.tests.util import smoketest


def test_plot_util():
    nruns = 4
    logdirs = [smoketest('--alg=ppo2 --env=CartPole-v0 --num_timesteps=10000') for _ in range(nruns)]
    data = pu.load_results(logdirs)
    assert len(data) == 4

    _, axes = pu.plot_results(data[:1]); assert len(axes) == 1
    _, axes = pu.plot_results(data, tiling='vertical'); assert axes.shape==(4,1)
    _, axes = pu.plot_results(data, tiling='horizontal'); assert axes.shape==(1,4)
    _, axes = pu.plot_results(data, tiling='symmetric'); assert axes.shape==(2,2)
    _, axes = pu.plot_results(data, split_fn=lambda _: ''); assert len(axes) == 1

