import tensorflow as tf

def contract_state_placeholder(contract, batch_size, name='ContrSt'):
    return tf.placeholder(shape=[batch_size], dtype=tf.int32, name=name)

def contract_state_input(contract, batch_size=None, name='ContrSt'):
    placeholder = contract_state_placeholder(contract, batch_size, name)
    return placeholder, tf.to_float(tf.one_hot(placeholder, contract.num_states))