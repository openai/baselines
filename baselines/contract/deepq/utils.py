from baselines.common.tf_util import adjust_shape
from baselines.contract.common.input import contract_state_input
from baselines.deepq.utils import ObservationInput
import tensorflow as tf
import numpy as np

class ContractStateAugmentedInput(ObservationInput):
    def __init__(self, observation_space, contracts, name=None):
        super().__init__(observation_space)
        self.contract_num_states = [c.num_states for c in contracts]
        self.contract_state_phs = [
            contract_state_input(c, name=c.name)[1] for c in contracts
        ]

    def get(self):
        return [
            super().get(),
        ] + self.contract_state_phs

    def make_feed_dict(self, data):
        """
        Assumes data is an interable whose second entry is an iterable of
        integer contract states.
        """
        assert data.shape[1] == 2, "Actual shape is: {}".format(data.shape)
        obs = np.array(data[0][0])
        contracts = np.array(data[0][1])

        feed_dict = super().make_feed_dict(obs)
        for i, ph in enumerate(self.contract_state_phs):
            c_one_hot = np.zeros([1, self.contract_num_states[i]])
            c_one_hot[0, contracts[i]] = 1
            feed_dict[ph] = c_one_hot
        return feed_dict
    
    def batch_size(self):
        return tf.shape(self.get()[0])[0]
