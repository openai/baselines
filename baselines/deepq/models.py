import tensorflow as tf
import tensorflow.contrib.layers as layers


def build_q_func(network, hiddens=[256], dueling=True, layer_norm=False, **network_kwargs):
    is_recurrent = None
    if isinstance(network, str):
        from baselines.common.models import get_network_builder
        network_builder_dict = get_network_builder(network)
        network = network_builder_dict['func'](**network_kwargs)
        is_recurrent = network_builder_dict['is_recurrent']

    def q_func_builder(input_placeholder, num_actions, scope, reuse=False):
        with tf.variable_scope(scope, reuse=reuse):
            if is_recurrent:
                raise NotImplementedError("DQN is not compatible with recurrent policies yet")

            policy_dict = network(input_placeholder, '', '', False)
            if not isinstance(policy_dict, dict):
                policy_dict = {'latent': policy_dict}
            if 'latent' in policy_dict:
                latent = policy_dict['latent']
            elif 'policy_latent' in policy_dict:
                latent = policy_dict['policy_latent']
            else:
                raise RuntimeError('DQN expects a network builder that supplies a latent space representation ("policy_latent" or "latent" entries filled in).')

            latent = layers.flatten(latent)

            with tf.variable_scope("action_value"):
                action_out = latent
                for hidden in hiddens:
                    action_out = layers.fully_connected(action_out, num_outputs=hidden, activation_fn=None)
                    if layer_norm:
                        action_out = layers.layer_norm(action_out, center=True, scale=True)
                    action_out = tf.nn.relu(action_out)
                action_scores = layers.fully_connected(action_out, num_outputs=num_actions, activation_fn=None)

            if dueling:
                with tf.variable_scope("state_value"):
                    state_out = latent
                    for hidden in hiddens:
                        state_out = layers.fully_connected(state_out, num_outputs=hidden, activation_fn=None)
                        if layer_norm:
                            state_out = layers.layer_norm(state_out, center=True, scale=True)
                        state_out = tf.nn.relu(state_out)
                    state_score = layers.fully_connected(state_out, num_outputs=1, activation_fn=None)
                action_scores_mean = tf.reduce_mean(action_scores, 1)
                action_scores_centered = action_scores - tf.expand_dims(action_scores_mean, 1)
                q_out = state_score + action_scores_centered
            else:
                q_out = action_scores
            return q_out

    return q_func_builder
