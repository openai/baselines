import numpy as np
import tensorflow as tf
from algos.a2c import A2C
from models.model import Model
from utils.distributions import make_pdtype
from utils.inputs import observation_input


class Mlp(A2C):
    def __init__(
            self,
            observation_space,
            action_space,
            nbatch,
            nsteps,
            reuse=False,
            name='Mlp'
    ):  # pylint: disable=W0613

        self.observation_space = observation_space
        self.action_space = action_space
        self.batch_size = nbatch
        self.steps = nsteps
        self.reuse = reuse
        self.name = name
        self.setup

    @Model.define_scope
    def setup(self):
        self.pdtype = make_pdtype(self.ation_space)
        with tf.variable_scope(self.name, reuse=self.reuse):
            X, processed_x = observation_input(
                self.observation_space,
                self.batch_size
            )
            processed_x = tf.layers.flatten(processed_x)
            pi_h1 = self.activation('tanh')(
                self.fc(processed_x, 'pi_fc1', nh=64, init_scale=np.sqrt(2))
            )
            pi_h2 = self.activation('tanh')(
                self.fc(pi_h1, 'pi_fc2', nh=64, init_scale=np.sqrt(2))
            )
            vf_h1 = self.activation('tanh')(
                self.fc(processed_x, 'vf_fc1', nh=64, init_scale=np.sqrt(2))
            )
            vf_h2 = self.activation('tanh')(
                self.fc(vf_h1, 'vf_fc2', nh=64, init_scale=np.sqrt(2))
            )
            vf = self.fc(vf_h2, 'vf', 1)[:, 0]

            self.pd, self.pi = self.pdtype.pdfromlatent(pi_h2, init_scale=0.01)

        a0 = self.pd.sample()
        neglogp0 = self.pd.neglogp(a0)
        self.initial_state = None
        self.X = X
        self.vf = vf
        self.a0 = a0
        self.neglogp0 = neglogp0

    def step(self, *_args, **_kwargs):
        a, v, neglogp = _kwargs['session'].run(
            [self.a0, self.vf, self.neglogp0],
            {self.X: _args[0]}
        )
        return a, v, self.initial_state, neglogp

    def value(self, *_args, **_kwargs):
        return _kwargs['session'].run(
            self.vf,
            {self.X: _args[0]}
        )
