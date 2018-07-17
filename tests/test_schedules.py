import numpy as np

from stable_baselines.common.schedules import ConstantSchedule, PiecewiseSchedule


def test_piecewise_schedule():
    """
    test PiecewiseSchedule
    """
    piecewise_sched = PiecewiseSchedule([(-5, 100), (5, 200), (10, 50), (100, 50), (200, -50)],
                                        outside_value=500)

    assert np.isclose(piecewise_sched.value(-10), 500)
    assert np.isclose(piecewise_sched.value(0), 150)
    assert np.isclose(piecewise_sched.value(5), 200)
    assert np.isclose(piecewise_sched.value(9), 80)
    assert np.isclose(piecewise_sched.value(50), 50)
    assert np.isclose(piecewise_sched.value(80), 50)
    assert np.isclose(piecewise_sched.value(150), 0)
    assert np.isclose(piecewise_sched.value(175), -25)
    assert np.isclose(piecewise_sched.value(201), 500)
    assert np.isclose(piecewise_sched.value(500), 500)

    assert np.isclose(piecewise_sched.value(200 - 1e-10), -50)


def test_constant_schedule():
    """
    test ConstantSchedule
    """
    constant_sched = ConstantSchedule(5)
    for i in range(-100, 100):
        assert np.isclose(constant_sched.value(i), 5)
