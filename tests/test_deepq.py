from baselines.deepq.experiments.custom_cartpole import main as main_custom
from baselines.deepq.experiments.train_cartpole import main as train_cartpole
from baselines.deepq.experiments.enjoy_cartpole import main as enjoy_cartpole
from baselines.deepq.experiments.train_mountaincar import main as train_mountaincar
from baselines.deepq.experiments.enjoy_mountaincar import main as enjoy_mountaincar


class DummyObject(object):
    """
    Dummy object to create fake Parsed Arguments object
    """
    pass


args = DummyObject()
args.no_render = True
args.max_timesteps = 200


def test_custom_cartpole():
<<<<<<< HEAD
    args = ['--no-render', '--max-timesteps', 1000]
    args = list(map(str, args))
    return_code = subprocess.call(['python', '-m', 'stable_baselines.deepq.experiments.custom_cartpole'] + args)
    _assert_eq(return_code, 0)

def test_cartpole():
    args = ['--max-timesteps', 1000]
    args = list(map(str, args))
    return_code = subprocess.call(['python', '-m', 'stable_baselines.deepq.experiments.train_cartpole'] + args)
    _assert_eq(return_code, 0)

    return_code = subprocess.call(['python', '-m', 'stable_baselines.deepq.experiments.enjoy_cartpole', '--no-render'])
    _assert_eq(return_code, 0)

def test_mountaincar():
    args = ['--max-timesteps', 1000]
    args = list(map(str, args))
    return_code = subprocess.call(['python', '-m', 'stable_baselines.deepq.experiments.train_mountaincar'] + args)
    _assert_eq(return_code, 0)

    return_code = subprocess.call(['python', '-m', 'stable_baselines.deepq.experiments.enjoy_mountaincar', '--no-render'])
    _assert_eq(return_code, 0)
=======
    main_custom(args)


def test_cartpole():
    train_cartpole(args)
    enjoy_cartpole(args)


def test_mountaincar():
    train_mountaincar(args)
    enjoy_mountaincar(args)
>>>>>>> refactoring
