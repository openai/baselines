import numpy as np


class AdaptiveParamNoiseSpec(object):
    """
    Implements adaptive parameter noise

    :param initial_stddev: (float) the initial value for the standard deviation of the noise
    :param desired_action_stddev: (float) the desired value for the standard deviation of the noise
    :param adoption_coefficient: (float) the update coefficient for the standard deviation of the noise
    """
    def __init__(self, initial_stddev=0.1, desired_action_stddev=0.1, adoption_coefficient=1.01):
        self.initial_stddev = initial_stddev
        self.desired_action_stddev = desired_action_stddev
        self.adoption_coefficient = adoption_coefficient

        self.current_stddev = initial_stddev

    def adapt(self, distance):
        """
        update the standard deviation for the parameter noise

        :param distance: (float) the noise distance applied to the parameters
        """
        if distance > self.desired_action_stddev:
            # Decrease stddev.
            self.current_stddev /= self.adoption_coefficient
        else:
            # Increase stddev.
            self.current_stddev *= self.adoption_coefficient

    def get_stats(self):
        """
        return the standard deviation for the parameter noise

        :return: (dict) the stats of the noise
        """
        return {'param_noise_stddev': self.current_stddev}

    def __repr__(self):
        fmt = 'AdaptiveParamNoiseSpec(initial_stddev={}, desired_action_stddev={}, adoption_coefficient={})'
        return fmt.format(self.initial_stddev, self.desired_action_stddev, self.adoption_coefficient)


class ActionNoise(object):
    """
    The action noise base class
    """
    def reset(self):
        """
        call end of episode reset for the noise
        """
        pass


class NormalActionNoise(ActionNoise):
    """
    A gaussian action noise

    :param mean: (float) the mean value of the noise
    :param sigma: (float) the scale of the noise (std here)
    """
    def __init__(self, mean, sigma):
        self._mu = mean
        self._sigma = sigma

    def __call__(self):
        return np.random.normal(self._mu, self._sigma)

    def __repr__(self):
        return 'NormalActionNoise(mu={}, sigma={})'.format(self._mu, self._sigma)


class OrnsteinUhlenbeckActionNoise(ActionNoise):
    """
    A Ornstein Uhlenbeck action noise, this is designed to aproximate brownian motion with friction.

    Based on http://math.stackexchange.com/questions/1287634/implementing-ornstein-uhlenbeck-in-matlab

    :param mean: (float) the mean of the noise
    :param sigma: (float) the scale of the noise
    :param theta: (float) the rate of mean reversion
    :param dt: (float) the timestep for the noise
    :param initial_noise: ([float]) the initial value for the noise output, (if None: 0)
    """

    def __init__(self, mean, sigma, theta=.15, dt=1e-2, initial_noise=None):
        self._theta = theta
        self._mu = mean
        self._sigma = sigma
        self._dt = dt
        self.initial_noise = initial_noise
        self.noise_prev = None
        self.reset()

    def __call__(self):
        noise = self.noise_prev + self._theta * (self._mu - self.noise_prev) * self._dt + \
                self._sigma * np.sqrt(self._dt) * np.random.normal(size=self._mu.shape)
        self.noise_prev = noise
        return noise

    def reset(self):
        """
        reset the Ornstein Uhlenbeck noise, to the initial position
        """
        self.noise_prev = self.initial_noise if self.initial_noise is not None else np.zeros_like(self._mu)

    def __repr__(self):
        return 'OrnsteinUhlenbeckActionNoise(mu={}, sigma={})'.format(self._mu, self._sigma)
