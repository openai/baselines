.. _custom_policy:

Custom Policy Network
---------------------

Stable baselines provides default policy networks for images (CNNPolicies)
and other type of input features (MlpPolicies).
However, you can also easily define a custom architecture for the policy (or value) network:

.. code-block:: python

  import gym

  from stable_baselines.common.policies import FeedForwardPolicy, register_policy
  from stable_baselines.common.vec_env import DummyVecEnv
  from stable_baselines import A2C

  # Custom MLP policy of three layers of size 128 each
  class CustomPolicy(FeedForwardPolicy):
      def __init__(self, *args, **kwargs):
          super(CustomPolicy, self).__init__(*args, **kwargs,
                                             layers=[128, 128, 128],
                                             feature_extraction="mlp")

  # Create and wrap the environment
  env = gym.make('LunarLander-v2')
  env = DummyVecEnv([lambda: env])

  model = A2C(CustomPolicy, env, verbose=1)
  # Train the agent
  model.learn(total_timesteps=100000)


You can also registered your policy, to help with code simplicity: you can refer to your custom policy using a string.

.. code-block:: python

  import gym

  from stable_baselines.common.policies import FeedForwardPolicy, register_policy
  from stable_baselines.common.vec_env import DummyVecEnv
  from stable_baselines import A2C

  # Custom MLP policy of three layers of size 128 each
  class CustomPolicy(FeedForwardPolicy):
      def __init__(self, *args, **kwargs):
          super(CustomPolicy, self).__init__(*args, **kwargs,
                                             layers=[128, 128, 128],
                                             feature_extraction="mlp")

  # Register the policy, it will check that the name is not already taken
  register_policy('CustomPolicy', CustomPolicy)

  # Because the policy is now registered, you can pass
  # a string to the agent constructor instead of passing a class
  model = A2C(policy='CustomPolicy', env='LunarLander-v2', verbose=1).learn(total_timesteps=100000)


If however, your task requires a more granular control over the policy architecture, you can redefine the policy directly:

.. code-block:: python

  import gym
  import tensorflow as tf

  from stable_baselines.common.policies import ActorCriticPolicy, register_policy, nature_cnn
  from stable_baselines.common.vec_env import DummyVecEnv
  from stable_baselines import A2C

  # Custom MLP policy of three layers of size 128 each for the actor and 2 layers of 32 for the critic,
  # with a nature_cnn feature extractor
  class CustomPolicy(ActorCriticPolicy):
      def __init__(self, sess, ob_space, ac_space, n_env, n_steps, n_batch, reuse=False, **kwargs):
          super(CustomPolicy, self).__init__(sess, ob_space, ac_space, n_env, n_steps, n_batch, n_lstm=256,
                                             reuse=reuse, scale=True)

          with tf.variable_scope("model", reuse=reuse):
              activ = tf.nn.relu

              extracted_features = nature_cnn(self.processed_x, **kwargs)
              extracted_features = tf.layers.flatten(extracted_features)

              pi_h = extracted_features
              for i, layer_size in enumerate([128, 128, 128]):
                  pi_h = activ(tf.layers.dense(pi_h, layer_size, name='pi_fc' + str(i)))
              pi_latent = pi_h

              vf_h = extracted_features
              for i, layer_size in enumerate([32, 32]):
                  vf_h = activ(tf.layers.dense(vf_h, layer_size, name='vf_fc' + str(i)))
              value_fn = tf.layers.dense(vf_h, 1, name='vf')
              vf_latent = vf_h

              self.proba_distribution, self.policy, self.q_value = \
                  self.pdtype.proba_distribution_from_latent(pi_latent, vf_latent, init_scale=0.01)

          self.value_fn = value_fn
          self.initial_state = None
          self._setup_init()

      def step(self, obs, state=None, mask=None):
          action, value, neglogp = self.sess.run([self.action, self._value, self.neglogp], {self.obs_ph: obs})
          return action, value, self.initial_state, neglogp

      def proba_step(self, obs, state=None, mask=None):
          return self.sess.run(self.policy_proba, {self.obs_ph: obs})

      def value(self, obs, state=None, mask=None):
          return self.sess.run(self._value, {self.obs_ph: obs})


  # Create and wrap the environment
  env = gym.make('Breakout-v0')
  env = DummyVecEnv([lambda: env])

  model = A2C(CustomPolicy, env, verbose=1)
  # Train the agent
  model.learn(total_timesteps=100000)
