import tensorflow as tf
import numpy as np
from baselines.common.models import fc, conv, conv_to_fc
import tensorflow.contrib.layers as layers


def augment_network_with_contract_state(base_network):
    def network(placeholders, **conv_kwargs):
        unscaled_images = placeholders[0]
        contracts = placeholders[1:]

        h_main = layers.flatten(base_network(unscaled_images, **conv_kwargs))
        h_cont = [
            fc(c, 'fc_' + str(i), nh=4, init_scale=np.sqrt(2)) for i, c in enumerate(contracts)
        ]
        return tf.concat([h_main] + h_cont, axis=-1)
    return network