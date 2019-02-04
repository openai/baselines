import tensorflow as tf
# from algos.a2c import A2C
from policies.agent import Agent
from utils.distributions import make_pdtype
from utils.inputs import observation_input


class Convnet(Agent):

    def __init__(
            self,
            observation_space,
            action_space,
            nbatch,
            nsteps,
            reuse=False,
            name='Convnet',
            **conv_kwargs
    ):
        ############################################################
        # set params                                               #
        ############################################################
        self.observation_space = observation_space
        self.action_space = action_space
        self.batch_size = nbatch
        self.steps = nsteps
        self.reuse = reuse
        self.name = name
        self.kwargs = conv_kwargs
        self.setup

    ################################################################
    # network architecture                                         #
    ################################################################
    @Agent.define_scope
    def setup(self):
        pdtype = make_pdtype(self.action_space)
        X, processed_x = observation_input(
            self.observation_space, self.batch_size
        )
        with tf.variable_scope(self.name, reuse=self.reuse):
            h = self.nature(processed_x, **self.kwargs)
            vf = self.dense(inputs=h, units=1, name='value_function',
                            kernel_initializer=self.ortho_init())[:, 0]
            pd, self.pi = pdtype.pdfromlatent(h, init_scale=0.01)

        a0 = pd.sample()
        neglogp0 = pd.neglogp(a0)
        self.initial_state = None

        self.X = X
        self.vf = vf
        self.a0 = a0
        self.neglogp0 = neglogp0
        self.pdtype = pdtype
        self.pd = pd

    # def step(self, *_args, **_kwargs):
    #     if _args and _kwargs:
    #         a, v, neglogp = _kwargs['session'].run(
    #             [self.a0, self.vf, self.neglogp0],
    #             feed_dict={self.X: _args[0]}
    #         )
    #         return a, v, self.initial_state, neglogp

    # def value(self, *_args, **_kwargs):
    #     if _args and _kwargs:
    #         return _kwargs['session'].run(
    #             self.vf,
    #             feed_dict={self.X: _args[0]}
    #         )

    def step(self, *args, **kwargs):
        step_fn = self.function(
            inputs=[self.X],
            outputs=[self.a0, self.vf, self.neglogp0]
        )
        a, v, neglogp = step_fn(kwargs['observations'])
        return a, v, self.initial_state, neglogp

    def value(self, *args, **kwargs):
        value_fn = self.function(inputs=[self.X], outputs=self.vf)
        return value_fn(kwargs['observations'])
