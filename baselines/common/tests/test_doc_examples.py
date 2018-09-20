import pytest
try:
    import mujoco_py
    _mujoco_present = True
except BaseException:
    mujoco_py = None
    _mujoco_present = False


@pytest.mark.skipif(
    not _mujoco_present,
    reason='error loading mujoco - either mujoco / mujoco key not present, or LD_LIBRARY_PATH is not pointing to mujoco library'
)
def test_lstm_example():
    import tensorflow as tf
    from baselines.common import policies, models, cmd_util
    from baselines.common.vec_env.dummy_vec_env import DummyVecEnv

    # create vectorized environment
    venv = DummyVecEnv([lambda: cmd_util.make_mujoco_env('Reacher-v2', seed=0)])

    with tf.Session() as sess:
        # build policy based on lstm network with 128 units
        policy = policies.build_policy(venv, models.lstm(128))(nbatch=1, nsteps=1)

        # initialize tensorflow variables
        sess.run(tf.global_variables_initializer())

        # prepare environment variables
        ob = venv.reset()
        state = policy.initial_state
        done = [False]
        step_counter = 0

        # run a single episode until the end (i.e. until done)
        while True:
            action, _, state, _ = policy.step(ob, S=state, M=done)
            ob, reward, done, _ = venv.step(action)
            step_counter += 1
            if done:
                break


        assert step_counter > 5




