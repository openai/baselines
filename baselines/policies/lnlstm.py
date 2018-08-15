import numpy as np
import tensorflow as tf
from algos.a2c import A2C
from policies.model import Model
from utils.distributions import make_pdtype
from utils.inputs import observation_input


class LnLstm(A2C):
    def __init__(
            self,
            observation_space,
            action_space,
            batch_size,
            nsteps,
            nlstm=256,
            reuse=False,
            name='lnstm'
    ):
        self.observation_space = observation_space
        self.action_space = action_space
        self.batch_size = batch_size
        self.steps = nsteps
        self.lstm_size = nlstm
        self.reuse = reuse
        self.name = name
        self.nenv = self.batch_size // self.steps
        self.setup

    @Model.define_scope
    def setup(self):
        self.pdtype = make_pdtype(self.action_space)
        X, processed_x = observation_input(
            self.observation_space, self.batch_size
        )
        M = tf.placeholder(tf.float32, [self.batch_size])  # mask (done t-1)
        S = tf.placeholder(tf.float32, [self.nenv, self.lstm_size * 2])  # states
        with tf.variable_scope(self.name, reuse=self.reuse):
            h = self.nature(processed_x)
            xs = self.batch_to_seq(h, self.nenv, self.steps)
            ms = self.batch_to_seq(M, self.nenv, self.steps)
            h5, snew = self.lnlstm(xs, ms, S, 'lstm1', nh=self.lstm_size)
            h5 = self.seq_to_batch(h5)
            vf = self.fc(h5, 'v', 1)
            self.pd, self.pi = self.pdtype.pdfromlatent(h5)

        v0 = vf[:, 0]
        a0 = self.pd.sample()
        neglogp0 = self.pd.neglogp(a0)
        self.initial_state = np.zeros((self.nenv, self.lstm_size * 2),
                                      dtype=np.float32)
        self.X = X
        self.M = M
        self.S = S
        self.snew = snew
        self.vf = vf
        self.v0 = v0
        self.a0 = a0
        self.neglogp0 = neglogp0

    def step(self, *_args, **_kwargs):
        return _kwargs['session'].run(
            [self.a0, self.v0, self.snew, self.neglogp0],
            {self.X: _kwargs['observations'],
             self.S: _kwargs['states'],
             self.M: _kwargs['masks']}
        )

    def value(self, *_args, **_kwargs):
        return _kwargs['session'].run(
            self.v0,
            {self.X: _kwargs['observations'],
             self.S: _kwargs['states'],
             self.M: _kwargs['masks']}
        )
