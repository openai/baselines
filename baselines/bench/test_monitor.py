from baselines.bench.monitor import Monitor
import gym
import json
import pandas
from datetime import datetime

def test_monitor():


    env_name = "AirRaidNoFrameskip-v4"
    env = gym.make(env_name)
    env.seed(0)
    mon_file = "baselines-test-{}-{}.monitor.csv".format(env_name,
                                                         datetime.now().strftime('%Y_%m%d_%H_%M'))
    menv = Monitor(env, mon_file)
    menv.reset()
    for _ in range(100000):
        _, _, done, _ = menv.step(env.action_space.sample())
        if done:
            menv.reset()

    f = open(mon_file, 'rt')

    firstline = f.readline()
    assert firstline.startswith('#')
    metadata = json.loads(firstline[1:])
    # assert metadata['env_id'] == "CartPole-v1"
    assert set(metadata.keys()) == {'env_id', 't_start'},  "Incorrect keys in monitor metadata"

    last_logline = pandas.read_csv(f, index_col=None)
    assert set(last_logline.keys()) == {'l', 't', 'r'}, "Incorrect keys in monitor logline"
    f.close()
    # os.remove(mon_file)

test_monitor()
