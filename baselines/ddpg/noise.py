import numpy as np


class AdaptiveParamNoiseSpec(object):
    def __init__(self, initial_stddev=0.1, desired_action_stddev=0.1, adoption_coefficient=1.01):
        """
        Implements adaptive parameter noise

        :param initial_stddev: (float) the initial value for the standard deviation of the noise
        :param desired_action_stddev: (float) the desired value for the standard deviation of the noise
        :param adoption_coefficient: (float) the update coefficient for the standard deviation of the noise
        """
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
    def __init__(self, mu, sigma):
        """
        A guassian action noise

        :param mu: (float) the position of the noise
        :param sigma: (float) the scale of the noise
        """
        self.mu = mu
        self.sigma = sigma

    def __call__(self):
        return np.random.normal(self.mu, self.sigma)

    def __repr__(self):
        return 'NormalActionNoise(mu={}, sigma={})'.format(self.mu, self.sigma)


class OrnsteinUhlenbeckActionNoise(ActionNoise):
    def __init__(self, mu, sigma, theta=.15, dt=1e-2, initial_noise=None):
        """
        A Ornstein Uhlenbeck action noise, this is designed to aproximate brownian motion with friction.

        Based on http://math.stackexchange.com/questions/1287634/implementing-ornstein-uhlenbeck-in-matlab

        :param mu: (float) the position of the noise
        :param sigma: (float) the scale of the noise
        :param theta: (float) the rate of mean reversion
        :param dt: (float) the timestep for the noise
        :param initial_noise: ([float]) the initial value for the noise output, (if None: 0)
        """
        self.theta = theta
        self.mu = mu
        self.sigma = sigma
        self.dt = dt
        self.initial_noise = initial_noise
        self.noise_prev = None
        self.reset()

    def __call__(self):
        noise = self.noise_prev + self.theta * (self.mu - self.noise_prev) * self.dt + \
                self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mu.shape)
        self.noise_prev = noise
        return noise

    def reset(self):
        """
        reset the Ornstein Uhlenbeck noise, to the initial position
        """
        self.noise_prev = self.initial_noise if self.initial_noise is not None else np.zeros_like(self.mu)

    def __repr__(self):
        return 'OrnsteinUhlenbeckActionNoise(mu={}, sigma={})'.format(self.mu, self.sigma)
