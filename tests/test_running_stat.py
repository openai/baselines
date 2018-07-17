import numpy as np

from stable_baselines.common.running_stat import RunningStat


def test_running_stat():
    """
    test RunningStat object
    """
    for shape in ((), (3,), (3, 4)):
        hist = []
        running_stat = RunningStat(shape)
        for _ in range(5):
            val = np.random.randn(*shape)
            running_stat.push(val)
            hist.append(val)
            _mean = np.mean(hist, axis=0)
            assert np.allclose(running_stat.mean, _mean)
            _var = np.square(_mean) if (len(hist) == 1) else np.var(hist, ddof=1, axis=0)
            assert np.allclose(running_stat.var, _var)
