import gym
import tensorflow as tf
import numpy as np
from baselines.common.vec_env.dummy_vec_env import DummyVecEnv
from baselines.common.tf_util import make_session
from baselines.ppo2.ppo2 import learn

def test_microbatches():
    def env_fn():
        env = gym.make('CartPole-v0')
        env.seed(0)
        return env

    env_ref = DummyVecEnv([env_fn])
    sess_ref = make_session(make_default=True, graph=tf.Graph())
    learn(env=env_ref, network='mlp', nsteps=32, total_timesteps=32, seed=0)
    vars_ref = {v.name: sess_ref.run(v) for v in tf.trainable_variables()}

    env_test = DummyVecEnv([env_fn])
    sess_test = make_session(make_default=True, graph=tf.Graph())
    learn(env=env_test, network='mlp', nsteps=32, total_timesteps=32, seed=0, microbatch_size=4)
    vars_test = {v.name: sess_test.run(v) for v in tf.trainable_variables()}

    for v in vars_ref:
        np.testing.assert_allclose(vars_ref[v], vars_test[v], atol=1e-3)

if __name__ == '__main__':
    test_microbatches()
