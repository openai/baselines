"""This file is used for specifying various schedules that evolve over
time throughout the execution of the algorithm, such as:

 - learning rate for the optimizer
 - exploration epsilon for the epsilon greedy exploration strategy
 - beta parameter for beta parameter in prioritized replay

Each schedule has a function `value(t)` which returns the current value
of the parameter given the timestep t of the optimization procedure.
"""


class Schedule(object):
    def value(self, step):
        """
        Value of the schedule for a given timestep

        :param step: (int) the timestep
        :return: (float) the output value for the given timestep
        """
        raise NotImplementedError


class ConstantSchedule(Schedule):
    """
    Value remains constant over time.

    :param value: (float) Constant value of the schedule
    """

    def __init__(self, value):
        self._value = value

    def value(self, step):
        return self._value


def linear_interpolation(left, right, alpha):
    """
    Linear interpolation between `left` and `right`.

    :param left: (float) left boundary
    :param right: (float) right boundary
    :param alpha: (float) coeff in [0, 1]
    :return: (float)
    """

    return left + alpha * (right - left)


class PiecewiseSchedule(Schedule):
    """
    Piecewise schedule.

    :param endpoints: ([(int, int)])
        list of pairs `(time, value)` meanining that schedule should output
        `value` when `t==time`. All the values for time must be sorted in
        an increasing order. When t is between two times, e.g. `(time_a, value_a)`
        and `(time_b, value_b)`, such that `time_a <= t < time_b` then value outputs
        `interpolation(value_a, value_b, alpha)` where alpha is a fraction of
        time passed between `time_a` and `time_b` for time `t`.
    :param interpolation: (lambda (float, float, float): float)
        a function that takes value to the left and to the right of t according
        to the `endpoints`. Alpha is the fraction of distance from left endpoint to
        right endpoint that t has covered. See linear_interpolation for example.
    :param outside_value: (float)
        if the value is requested outside of all the intervals sepecified in
        `endpoints` this value is returned. If None then AssertionError is
        raised when outside value is requested.
    """

    def __init__(self, endpoints, interpolation=linear_interpolation, outside_value=None):
        idxes = [e[0] for e in endpoints]
        assert idxes == sorted(idxes)
        self._interpolation = interpolation
        self._outside_value = outside_value
        self._endpoints = endpoints

    def value(self, step):
        for (left_t, left), (right_t, right) in zip(self._endpoints[:-1], self._endpoints[1:]):
            if left_t <= step < right_t:
                alpha = float(step - left_t) / (right_t - left_t)
                return self._interpolation(left, right, alpha)

        # t does not belong to any of the pieces, so doom.
        assert self._outside_value is not None
        return self._outside_value


class LinearSchedule(Schedule):
    """
    Linear interpolation between initial_p and final_p over
    schedule_timesteps. After this many timesteps pass final_p is
    returned.

    :param schedule_timesteps: (int) Number of timesteps for which to linearly anneal initial_p to final_p
    :param initial_p: (float) initial output value
    :param final_p: (float) final output value
    """

    def __init__(self, schedule_timesteps, final_p, initial_p=1.0):
        self.schedule_timesteps = schedule_timesteps
        self.final_p = final_p
        self.initial_p = initial_p

    def value(self, step):
        fraction = min(float(step) / self.schedule_timesteps, 1.0)
        return self.initial_p + fraction * (self.final_p - self.initial_p)
