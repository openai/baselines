.. _custom_policy:

Custom Policy Network
---------------------

Stable baselines provides default policy networks (see :ref:`Policies <policies>` ) for images (CNNPolicies)
and other type of input features (MlpPolicies).

One way of customising the policy network architecture is to pass arguments when creating the model,
using ``policy_kwargs`` parameter:

.. code-block:: python

  import gym
  import tensorflow as tf

  from stable_baselines import PPO2

  # Custom MLP policy of two layers of size 32 each with tanh activation function
  policy_kwargs = dict(act_fun=tf.nn.tanh, net_arch=[32, 32])
  # Create the agent
  model = PPO2("MlpPolicy", "CartPole-v1", policy_kwargs=policy_kwargs, verbose=1)
  # Retrieve the environment
  env = model.get_env()
  # Train the agent
  model.learn(total_timesteps=100000)
  # Save the agent
  model.save("ppo2-cartpole")

  del model
  # the policy_kwargs are automatically loaded
  model = PPO2.load("ppo2-cartpole")


You can also easily define a custom architecture for the policy (or value) network:

.. note::

    Defining a custom policy class is equivalent to passing ``policy_kwargs``.
    However, it lets you name the policy and so makes usually the code clearer.
    ``policy_kwargs`` should be rather used when doing hyperparameter search.



.. code-block:: python

  import gym

  from stable_baselines.common.policies import FeedForwardPolicy, register_policy
  from stable_baselines.common.vec_env import DummyVecEnv
  from stable_baselines import A2C

  # Custom MLP policy of three layers of size 128 each
  class CustomPolicy(FeedForwardPolicy):
      def __init__(self, *args, **kwargs):
          super(CustomPolicy, self).__init__(*args, **kwargs,
                                             net_arch=[dict(pi=[128, 128, 128],
                                                            vf=[128, 128, 128])],
                                             feature_extraction="mlp")

  # Create and wrap the environment
  env = gym.make('LunarLander-v2')
  env = DummyVecEnv([lambda: env])

  model = A2C(CustomPolicy, env, verbose=1)
  # Train the agent
  model.learn(total_timesteps=100000)
  # Save the agent
  model.save("a2c-lunar")

  del model
  # When loading a model with a custom policy
  # you MUST pass explicitly the policy when loading the saved model
  model = A2C.load("a2c-lunar", policy=CustomPolicy)

.. warning::

    When loading a model with a custom policy, you must pass the custom policy explicitly when loading the model.
    (cf previous example)


You can also register your policy, to help with code simplicity: you can refer to your custom policy using a string.

.. code-block:: python

  import gym

  from stable_baselines.common.policies import FeedForwardPolicy, register_policy
  from stable_baselines.common.vec_env import DummyVecEnv
  from stable_baselines import A2C

  # Custom MLP policy of three layers of size 128 each
  class CustomPolicy(FeedForwardPolicy):
      def __init__(self, *args, **kwargs):
          super(CustomPolicy, self).__init__(*args, **kwargs,
                                             net_arch=[dict(pi=[128, 128, 128],
                                                            vf=[128, 128, 128])],
                                             feature_extraction="mlp")

  # Register the policy, it will check that the name is not already taken
  register_policy('CustomPolicy', CustomPolicy)

  # Because the policy is now registered, you can pass
  # a string to the agent constructor instead of passing a class
  model = A2C(policy='CustomPolicy', env='LunarLander-v2', verbose=1).learn(total_timesteps=100000)


.. deprecated:: 2.3.0

  Use ``net_arch`` instead of ``layers`` parameter to define the network architecture. It allows to have a greater control.


The ``net_arch`` parameter of ``FeedForwardPolicy`` allows to specify the amount and size of the hidden layers and how many
of them are shared between the policy network and the value network. It is assumed to be a list with the following
structure:

1. An arbitrary length (zero allowed) number of integers each specifying the number of units in a shared layer.
   If the number of ints is zero, there will be no shared layers.
2. An optional dict, to specify the following non-shared layers for the value network and the policy network.
   It is formatted like ``dict(vf=[<value layer sizes>], pi=[<policy layer sizes>])``.
   If it is missing any of the keys (pi or vf), no non-shared layers (empty list) is assumed.

In short: ``[<shared layers>, dict(vf=[<non-shared value network layers>], pi=[<non-shared policy network layers>])]``.

Examples
~~~~~~~~

Two shared layers of size 128: ``net_arch=[128, 128]``


.. code-block:: none

                  obs
                   |
                 <128>
                   |
                 <128>
           /               \
        action            value


Value network deeper than policy network, first layer shared: ``net_arch=[128, dict(vf=[256, 256])]``

.. code-block:: none

                  obs
                   |
                 <128>
           /               \
        action             <256>
                             |
                           <256>
                             |
                           value


Initially shared then diverging: ``[128, dict(vf=[256], pi=[16])]``

.. code-block:: none

                  obs
                   |
                 <128>
           /               \
         <16>             <256>
           |                |
        action            value

The ``LstmPolicy`` can be used to construct recurrent policies in a similar way:

.. code-block:: python

    class CustomLSTMPolicy(LstmPolicy):
        def __init__(self, sess, ob_space, ac_space, n_env, n_steps, n_batch, n_lstm=64, reuse=False, **_kwargs):
            super().__init__(sess, ob_space, ac_space, n_env, n_steps, n_batch, n_lstm, reuse,
                             net_arch=[8, 'lstm', dict(vf=[5, 10], pi=[10])],
                             layer_norm=True, feature_extraction="mlp", **_kwargs)

Here the ``net_arch`` parameter takes an additional (mandatory) 'lstm' entry within the shared network section.
The LSTM is shared between value network and policy network.




If your task requires even more granular control over the policy architecture, you can redefine the policy directly:

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
          super(CustomPolicy, self).__init__(sess, ob_space, ac_space, n_env, n_steps, n_batch, reuse=reuse, scale=True)

          with tf.variable_scope("model", reuse=reuse):
              activ = tf.nn.relu

              extracted_features = nature_cnn(self.processed_obs, **kwargs)
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

      def step(self, obs, state=None, mask=None, deterministic=False):
          if deterministic:
              action, value, neglogp = self.sess.run([self.deterministic_action, self._value, self.neglogp],
                                                     {self.obs_ph: obs})
          else:
              action, value, neglogp = self.sess.run([self.action, self._value, self.neglogp],
                                                     {self.obs_ph: obs})
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
