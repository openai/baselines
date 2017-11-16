import os.path as osp

_atari7 = ['BeamRider', 'Breakout', 'Enduro', 'Pong', 'Qbert', 'Seaquest', 'SpaceInvaders']
_atariexpl7 = ['Freeway', 'Gravitar', 'MontezumaRevenge', 'Pitfall', 'PrivateEye', 'Solaris', 'Venture']

_BENCHMARKS = []


def register_benchmark(benchmark):
    for b in _BENCHMARKS:
        if b['name'] == benchmark['name']:
            raise ValueError('Benchmark with name %s already registered!' % b['name'])
    _BENCHMARKS.append(benchmark)


def list_benchmarks():
    return [b['name'] for b in _BENCHMARKS]


def get_benchmark(benchmark_name):
    for b in _BENCHMARKS:
        if b['name'] == benchmark_name:
            return b
    raise ValueError('%s not found! Known benchmarks: %s' % (benchmark_name, list_benchmarks()))


def get_task(benchmark, env_id):
    """Get a task by env_id. Return None if the benchmark doesn't have the env"""
    return next(filter(lambda task: task['env_id'] == env_id, benchmark['tasks']), None)


def find_task_for_env_id_in_any_benchmark(env_id):
    for bm in _BENCHMARKS:
        for task in bm["tasks"]:
            if task["env_id"] == env_id:
                return bm, task
    return None, None


_ATARI_SUFFIX = 'NoFrameskip-v4'

register_benchmark({
    'name': 'Atari50M',
    'description': '7 Atari games from Mnih et al. (2013), with pixel observations, 50M timesteps',
    'tasks': [{'env_id': _game + _ATARI_SUFFIX, 'trials': 2, 'num_timesteps': int(50e6)} for _game in _atari7]
})

register_benchmark({
    'name': 'Atari10M',
    'description': '7 Atari games from Mnih et al. (2013), with pixel observations, 10M timesteps',
    'tasks': [{'env_id': _game + _ATARI_SUFFIX, 'trials': 2, 'num_timesteps': int(10e6)} for _game in _atari7]
})

register_benchmark({
    'name': 'Atari1Hr',
    'description': '7 Atari games from Mnih et al. (2013), with pixel observations, 1 hour of walltime',
    'tasks': [{'env_id': _game + _ATARI_SUFFIX, 'trials': 2, 'num_seconds': 60 * 60} for _game in _atari7]
})

register_benchmark({
    'name': 'AtariExploration10M',
    'description': '7 Atari games emphasizing exploration, with pixel observations, 10M timesteps',
    'tasks': [{'env_id': _game + _ATARI_SUFFIX, 'trials': 2, 'num_timesteps': int(10e6)} for _game in _atariexpl7]
})




# MuJoCo

_mujocosmall = [
    'InvertedDoublePendulum-v1', 'InvertedPendulum-v1',
    'HalfCheetah-v1', 'Hopper-v1', 'Walker2d-v1',
    'Reacher-v1', 'Swimmer-v1']
register_benchmark({
    'name': 'Mujoco1M',
    'description': 'Some small 2D MuJoCo tasks, run for 1M timesteps',
    'tasks': [{'env_id': _envid, 'trials': 3, 'num_timesteps': int(1e6)} for _envid in _mujocosmall]
})
register_benchmark({
    'name': 'MujocoWalkers',
    'description': 'MuJoCo forward walkers, run for 8M, humanoid 100M',
    'tasks': [
        {'env_id': "Hopper-v1", 'trials': 4, 'num_timesteps': 8 * 1000000},
        {'env_id': "Walker2d-v1", 'trials': 4, 'num_timesteps': 8 * 1000000},
        {'env_id': "Humanoid-v1", 'trials': 4, 'num_timesteps': 100 * 1000000},
    ]
})

# Roboschool

register_benchmark({
    'name': 'Roboschool8M',
    'description': 'Small 2D tasks, up to 30 minutes to complete on 8 cores',
    'tasks': [
        {'env_id': "RoboschoolReacher-v1", 'trials': 4, 'num_timesteps': 2 * 1000000},
        {'env_id': "RoboschoolAnt-v1", 'trials': 4, 'num_timesteps': 8 * 1000000},
        {'env_id': "RoboschoolHalfCheetah-v1", 'trials': 4, 'num_timesteps': 8 * 1000000},
        {'env_id': "RoboschoolHopper-v1", 'trials': 4, 'num_timesteps': 8 * 1000000},
        {'env_id': "RoboschoolWalker2d-v1", 'trials': 4, 'num_timesteps': 8 * 1000000},
    ]
})
register_benchmark({
    'name': 'RoboschoolHarder',
    'description': 'Test your might!!! Up to 12 hours on 32 cores',
    'tasks': [
        {'env_id': "RoboschoolHumanoid-v1", 'trials': 4, 'num_timesteps': 100 * 1000000},
        {'env_id': "RoboschoolHumanoidFlagrun-v1", 'trials': 4, 'num_timesteps': 200 * 1000000},
        {'env_id': "RoboschoolHumanoidFlagrunHarder-v1", 'trials': 4, 'num_timesteps': 400 * 1000000},
    ]
})

# Other

_atari50 = [  # actually 47
    'Alien', 'Amidar', 'Assault', 'Asterix', 'Asteroids',
    'Atlantis', 'BankHeist', 'BattleZone', 'BeamRider', 'Bowling',
    'Breakout', 'Centipede', 'ChopperCommand', 'CrazyClimber',
    'DemonAttack', 'DoubleDunk', 'Enduro', 'FishingDerby', 'Freeway',
    'Frostbite', 'Gopher', 'Gravitar', 'IceHockey', 'Jamesbond',
    'Kangaroo', 'Krull', 'KungFuMaster', 'MontezumaRevenge', 'MsPacman',
    'NameThisGame', 'Pitfall', 'Pong', 'PrivateEye', 'Qbert',
    'RoadRunner', 'Robotank', 'Seaquest', 'SpaceInvaders', 'StarGunner',
    'Tennis', 'TimePilot', 'Tutankham', 'UpNDown', 'Venture',
    'VideoPinball', 'WizardOfWor', 'Zaxxon',
]

register_benchmark({
    'name': 'Atari50_10M',
    'description': '47 Atari games from Mnih et al. (2013), with pixel observations, 10M timesteps',
    'tasks': [{'env_id': _game + _ATARI_SUFFIX, 'trials': 2, 'num_timesteps': int(10e6)} for _game in _atari50]
})
