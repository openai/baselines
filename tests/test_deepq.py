from stable_baselines.deepq.experiments.custom_cartpole import main as main_custom
from stable_baselines.deepq.experiments.train_cartpole import main as train_cartpole
from stable_baselines.deepq.experiments.enjoy_cartpole import main as enjoy_cartpole
from stable_baselines.deepq.experiments.train_mountaincar import main as train_mountaincar
from stable_baselines.deepq.experiments.enjoy_mountaincar import main as enjoy_mountaincar


class DummyObject(object):
    """
    Dummy object to create fake Parsed Arguments object
    """
    pass


args = DummyObject()
args.no_render = True
args.max_timesteps = 200


def test_custom_cartpole():
    main_custom(args)


def test_cartpole():
    train_cartpole(args)
    enjoy_cartpole(args)


def test_mountaincar():
    train_mountaincar(args)
    enjoy_mountaincar(args)
