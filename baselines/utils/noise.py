# coding: utf-8

import numpy as np


class AdaptiveParamNoiseSpec(object):
    def __init__(
            self,
            initial_stddev=0.1,
            desired_action_stddev=0.1,
            adoption_coefficient=1.01
    ):
        self.initial_stddev = initial_stddev
        self.desired_action_stddev = desired_action_stddev
        self.adoption_coefficient = adoption_coefficient

        self.current_stddev = initial_stddev

    def adapt(self, distance):
        if distance > self.desired_action_stddev:
            # Decrease stddev.
            self.current_stddev /= self.adoption_coefficient
        else:
            # Increase stddev.
            self.current_stddev *= self.adoption_coefficient

    def get_stats(self):
        stats = {
            'param_noise_stddev': self.current_stddev,
        }
        return stats

    def __repr__(self):
        fmt = 'AdaptiveParamNoiseSpec(initial_stddev={}, \
         desired_action_stddev={}, adoption_coefficient={})'
        return fmt.format(self.initial_stddev, self.desired_action_stddev,
                          self.adoption_coefficient)


class ActionNoise(object):
    def reset(self):
        pass


class NormalActionNoise(ActionNoise):
    def __init__(self, μ, σ):
        self.μ = μ
        self.σ = σ

    def __call__(self):
        return np.random.normal(self.μ, self.σ)

    def __repr__(self):
        return 'NormalActionNoise(μ={}, σ={})'.format(self.μ, self.σ)


# Based on http://math.stackexchange.com/questions/1287634/implementing-ornstein-uhlenbeck-in-matlab
class OrnsteinUhlenbeckActionNoise(ActionNoise):
    def __init__(self, μ, σ, θ=.15, dt=1e-2, x0=None):
        self.θ = θ
        self.μ = μ
        self.σ = σ
        self.dt = dt
        self.x0 = x0
        self.reset()

    def __call__(self):
        x = self.x_prev + self.θ * (self.μ - self.x_prev) * self.dt + \
            self.σ * np.sqrt(self.dt) * np.random.normal(size=self.μ.shape)
        self.x_prev = x
        return x

    def reset(self):
        self.x_prev = self.x0 if self.x0 is not None else np.zeros_like(self.μ)

    def __repr__(self):
        return 'OrnsteinUhlenbeckActionNoise(μ={}, σ={})'.format(self.μ, self.σ)
