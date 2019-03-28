import uuid
import json
import os

import pandas
import gym

from stable_baselines.bench import Monitor


def test_monitor():
    """
    test the monitor wrapper
    """
    env = gym.make("CartPole-v1")
    env.seed(0)
    mon_file = "/tmp/stable_baselines-test-{}.monitor.csv".format(uuid.uuid4())
    menv = Monitor(env, mon_file)
    menv.reset()
    for _ in range(1000):
        _, _, done, _ = menv.step(0)
        if done:
            menv.reset()

    file_handler = open(mon_file, 'rt')

    firstline = file_handler.readline()
    assert firstline.startswith('#')
    metadata = json.loads(firstline[1:])
    assert metadata['env_id'] == "CartPole-v1"
    assert set(metadata.keys()) == {'env_id', 't_start'}, "Incorrect keys in monitor metadata"

    last_logline = pandas.read_csv(file_handler, index_col=None)
    assert set(last_logline.keys()) == {'l', 't', 'r'}, "Incorrect keys in monitor logline"
    file_handler.close()
    os.remove(mon_file)
