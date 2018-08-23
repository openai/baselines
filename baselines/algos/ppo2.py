import tensorflow as tf
from policies.agent import Agent


class PPO2(Agent):
    def __init__(
            self,
            policy,
            observation_space,
            action_space,
            nbatch_act,
            nbatch_train,
            nsteps,
            ent_coef,
            vf_coef,
            max_grad_norm
    ):
        super(PPO2, self).__init__(name='PPO2')

        ############################################################
        # set params                                               #
        ############################################################
        self.vf_coef = vf_coef
        self.ent_coef = ent_coef
        self.act_model = policy(
            observation_space,
            action_space,
            nbatch_act,
            1,
            name='PPO2',
            reuse=False
        )
        self.train_model = policy(
            observation_space,
            action_space,
            nbatch_train,
            nsteps,
            name='PPO2',
            reuse=True
        )
        self.action = self.train_model.pdtype.sample_placeholder([None])
        self.advantage = tf.placeholder(tf.float32, [None])
        self.reward = tf.placeholder(tf.float32, [None])
        self.old_neglogpac = tf.placeholder(tf.float32, [None])
        self.old_vpred = tf.placeholder(tf.float32, [None])
        self.learning_rate = tf.placeholder(tf.float32, [])
        self.cliprange = tf.placeholder(tf.float32, [])
        self.max_grad_norm = max_grad_norm
        self.loss_names = ['policy_loss',
                           'value_loss',
                           'policy_entropy',
                           'approxkl',
                           'clipfrac']
        self.step = self.act_model.step
        self.value = self.act_model.value
        self.initial_state = self.act_model.initial_state
        self.loss
        self.optimize

    @Agent.define_scope
    def loss(self):
        neglogpac = self.train_model.pd.neglogp(self.action)
        self.entropy = tf.reduce_mean(self.train_model.pd.entropy())

        vpred = self.train_model.vf
        vpredclipped = self.old_vpred + tf.clip_by_value(
            self.train_model.vf - self.old_vpred,
            - self.cliprange,
            self.cliprange
        )
        vf_losses1 = tf.square(vpred - self.reward)
        vf_losses2 = tf.square(vpredclipped - self.reward)
        self.vf_loss = .5 * tf.reduce_mean(tf.maximum(vf_losses1, vf_losses2))
        ratio = tf.exp(self.old_neglogpac - neglogpac)
        pg_losses = -self.advantage * ratio
        pg_losses2 = -self.advantage * tf.clip_by_value(
            ratio,
            1.0 - self.cliprange,
            1.0 + self.cliprange
        )
        self.pg_loss = tf.reduce_mean(tf.maximum(pg_losses, pg_losses2))
        self.approxkl = .5 * tf.reduce_mean(
            tf.square(neglogpac - self.old_neglogpac)
        )
        self.clipfrac = tf.reduce_mean(
            tf.to_float(tf.greater(tf.abs(ratio - 1.0), self.cliprange))
        )
        loss = self.pg_loss - self.entropy * self.ent_coef \
            + self.vf_loss * self.vf_coef
        return loss

    @Agent.define_scope
    def optimize(self):
        with tf.variable_scope('PPO2'):
            params = tf.trainable_variables()
        grads = tf.gradients(self.loss, params)
        if self.max_grad_norm is not None:
            grads, _grad_norm = tf.clip_by_global_norm(grads, self.max_grad_norm)
        grads = list(zip(grads, params))
        trainer = tf.train.AdamOptimizer(learning_rate=self.learning_rate,
                                         epsilon=1e-5)
        _train = trainer.apply_gradients(grads)
        return _train

    def predict(
            self,
            lr,
            cliprange,
            obs,
            returns,
            masks,
            actions,
            values,
            neglogpacs,
            states=None
    ):
        advs = returns - values
        advs = (advs - advs.mean()) / (advs.std() + 1e-8)
        td_map = {
            self.train_model.X: obs,
            self.action: actions,
            self.advantage: advs,
            self.reward: returns,
            self.learning_rate: lr,
            self.cliprange: cliprange,
            self.old_neglogpac: neglogpacs,
            self.old_vpred: values
        }
        if states is not None:
            td_map[self.train_model.S] = states
            td_map[self.train_model.M] = masks
        return self.function(
            inputs=tuple(td_map.keys()),
            outputs=[self.pg_loss,
                     self.vf_loss,
                     self.entropy,
                     self.approxkl,
                     self.clipfrac,
                     self.optimize],
            givens=td_map
        )()[:-1]
