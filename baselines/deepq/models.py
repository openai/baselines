import tensorflow as tf


def build_q_func(network, hiddens=[256], dueling=True, layer_norm=False, **network_kwargs):
    if isinstance(network, str):
        from baselines.common.models import get_network_builder
        network = get_network_builder(network)(**network_kwargs)

    def q_func_builder(input_shape, num_actions):
        # the sub Functional model which does not include the top layer.
        model = network(input_shape)

        # wrapping the sub Functional model with layers that compute action scores into another Functional model.
        latent = model.outputs
        if len(latent) > 1:
            if latent[1] is not None:
                raise NotImplementedError("DQN is not compatible with recurrent policies yet")
        latent = latent[0]

        latent = tf.keras.layers.Flatten()(latent)

        with tf.name_scope("action_value"):
            action_out = latent
            for hidden in hiddens:
                action_out = tf.keras.layers.Dense(units=hidden, activation=None)(action_out)
                if layer_norm:
                    action_out = tf.keras.layers.LayerNormalization(center=True, scale=True)(action_out)
                action_out = tf.nn.relu(action_out)
            action_scores = tf.keras.layers.Dense(units=num_actions, activation=None)(action_out)

        if dueling:
            with tf.name_scope("state_value"):
                state_out = latent
                for hidden in hiddens:
                    state_out = tf.keras.layers.Dense(units=hidden, activation=None)(state_out)
                    if layer_norm:
                        state_out = tf.keras.layers.LayerNormalization(center=True, scale=True)(state_out)
                    state_out = tf.nn.relu(state_out)
                state_score = tf.keras.layers.Dense(units=1, activation=None)(state_out)
            action_scores_mean = tf.reduce_mean(action_scores, 1)
            action_scores_centered = action_scores - tf.expand_dims(action_scores_mean, 1)
            q_out = state_score + action_scores_centered
        else:
            q_out = action_scores
        return tf.keras.Model(inputs=model.inputs, outputs=[q_out])

    return q_func_builder
