import tensorflow as tf
from policies.model import Model
from utils.schedules import Scheduler


class A2C(Model):

    def __init__(
            self,
            policy,
            observation_space,
            action_space,
            nenvs,
            nsteps,
            ent_coef=0.01,
            vf_coef=0.5,
            max_grad_norm=0.5,
            lr=7e-4,
            alpha=0.99,
            epsilon=1e-5,
            total_timesteps=int(80e6),
            lrschedule='linear'
    ):
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm
        self.lr = lr
        self.alpha = alpha
        self.epsilon = epsilon
        self.total_timesteps = total_timesteps
        self.lrschedule = lrschedule
        super(A2C, self).__init__(name='A2C')

        ############################################################
        # Set params                                               #
        ############################################################
        nbatch = nenvs * nsteps

        tf.reset_default_graph()
        self.action = tf.placeholder(tf.int32, [nbatch], name='action')  # Action
        self.advantage = tf.placeholder(tf.float32, [nbatch], name='advantage')  # Advantage
        self.reward = tf.placeholder(tf.float32, [nbatch], name='reward')  # Reward
        self.learning_rate = tf.placeholder(tf.float32, [], name='learning_rate')  # Learning Rate

        self.step_model = policy(
            observation_space,
            action_space,
            nenvs,
            1,
            reuse=False,
            name='A2C'
        )

        self.train_model = policy(
            observation_space,
            action_space,
            nenvs*nsteps,
            nsteps,
            reuse=True,
            name='A2C'
        )
        # import pdb; pdb.set_trace() ## DEBUG ##
        self.loss
        self.optimizer

    ################################################################
    # Loss function                                                #
    ################################################################
    @Model.define_scope
    def loss(self):
        neglogpac = tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=self.train_model.pi, labels=self.action
        )
        # policy gradient loss
        self.policy_loss = tf.reduce_mean(self.advantage * neglogpac)
        self.value_loss = tf.reduce_mean(
            tf.losses.mean_squared_error(
                tf.squeeze(self.train_model.vf), self.reward
            )
        )
        # value function loss
        self.entropy = tf.reduce_mean(
            self.categorical_entropy(self.train_model.pi)
        )
        loss = self.policy_loss - self.entropy * self.ent_coef + \
            self.value_loss * self.vf_coef

        return loss

    ################################################################
    # Optimizer                                                    #
    ################################################################
    @Model.define_scope
    def optimizer(self):
        with tf.variable_scope('A2C'):
            self.params = tf.trainable_variables()

        grads = tf.gradients(self.loss, self.params)
        if self.max_grad_norm is not None:
            grads, grad_norm = tf.clip_by_global_norm(
                grads,
                self.max_grad_norm
            )

        grads = list(zip(grads, self.params))
        trainer = tf.train.RMSPropOptimizer(
            learning_rate=self.learning_rate,
            decay=self.alpha,
            epsilon=self.epsilon
        )
        self._train = trainer.apply_gradients(grads)
        self.scheduled_lr = Scheduler(
            v=self.lr,
            nvalues=self.total_timesteps,
            schedule=self.lrschedule
        )
        self.initial_state = self.step_model.initial_state

    ################################################################
    # Inference                                                    #
    ################################################################
    def predict(self, *_args, **_kwargs):
        # with tf.variable_scope('_cache_predict'):
        if _kwargs:
            advs = _kwargs['rewards'] - _kwargs['values']

            for step in range(len(_kwargs['observations'])):
                cur_lr = self.scheduled_lr.value()

            td_map = {
                self.train_model.X: _kwargs['observations'],
                self.action: _kwargs['actions'],
                self.advantage: advs,
                self.reward: _kwargs['rewards'],
                self.learning_rate: cur_lr
            }

            if _kwargs['states'] is not None:
                td_map[self.train_model.S] = _kwargs['states']
                td_map[self.train_model.M] = _kwargs['masks']

            policy_loss, value_loss, policy_entropy, _=_kwargs['session'].run(
                [self.policy_loss,
                 self.value_loss,
                 self.entropy,
                 self._train],
                feed_dict=td_map
            )
            return policy_loss, value_loss, policy_entropy
