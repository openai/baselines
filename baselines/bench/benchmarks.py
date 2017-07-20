_atari7 = ['BeamRider', 'Breakout', 'Enduro', 'Pong', 'Qbert', 'Seaquest', 'SpaceInvaders']
_atariexpl7 = ['Freeway', 'Gravitar', 'MontezumaRevenge', 'Pitfall', 'PrivateEye', 'Solaris', 'Venture']

_BENCHMARKS = []

def register_benchmark(benchmark):
    for b in _BENCHMARKS:
        if b['name'] == benchmark['name']:
            raise ValueError('Benchmark with name %s already registered!'%b['name'])
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

_ATARI_SUFFIX = 'NoFrameskip-v4'

register_benchmark({
    'name' : 'Atari200M',
    'description' :'7 Atari games from Mnih et al. (2013), with pixel observations, 200M frames',
    'tasks'  : [{'env_id' : _game + _ATARI_SUFFIX, 'trials' : 2, 'num_timesteps' : int(200e6)} for _game in _atari7]
})

register_benchmark({
    'name' : 'Atari40M',
    'description' :'7 Atari games from Mnih et al. (2013), with pixel observations, 40M frames',
    'tasks'  : [{'env_id' : _game + _ATARI_SUFFIX, 'trials' : 2, 'num_timesteps' : int(40e6)} for _game in _atari7]
})

register_benchmark({
    'name' : 'Atari1Hr',
    'description' :'7 Atari games from Mnih et al. (2013), with pixel observations, 1 hour of walltime',
    'tasks'  : [{'env_id' : _game + _ATARI_SUFFIX, 'trials' : 2, 'num_seconds' : 60*60} for _game in _atari7]
})

register_benchmark({
    'name' : 'AtariExploration40M',
    'description' :'7 Atari games emphasizing exploration, with pixel observations, 40M frames',
    'tasks'  : [{'env_id' : _game + _ATARI_SUFFIX, 'trials' : 2, 'num_timesteps' : int(40e6)} for _game in _atariexpl7]
})


_mujocosmall = [
    'InvertedDoublePendulum-v1', 'InvertedPendulum-v1',
    'HalfCheetah-v1', 'Hopper-v1', 'Walker2d-v1',
    'Reacher-v1', 'Swimmer-v1']

register_benchmark({
    'name' : 'Mujoco1M',
    'description' : 'Some small 2D MuJoCo tasks, run for 1M timesteps',
    'tasks' : [{'env_id' : _envid, 'trials' : 3, 'num_timesteps' : int(1e6)} for _envid in _mujocosmall]
})

_roboschool_mujoco = [
    'RoboschoolInvertedDoublePendulum-v0', 'RoboschoolInvertedPendulum-v0',      # cartpole
    'RoboschoolHalfCheetah-v0', 'RoboschoolHopper-v0', 'RoboschoolWalker2d-v0',  # forward walkers
    'RoboschoolReacher-v0'
    ]

register_benchmark({
    'name' : 'RoboschoolMujoco2M',
    'description' : 'Same small 2D tasks, still improving up to 2M',
    'tasks' : [{'env_id' : _envid, 'trials' : 3, 'num_timesteps' : int(2e6)} for _envid in _roboschool_mujoco]
})


_atari50 =  [ # actually 49
            'Alien', 'Amidar', 'Assault', 'Asterix', 'Asteroids', 
            'Atlantis', 'BankHeist', 'BattleZone', 'BeamRider',  'Bowling', 
            'Boxing', 'Breakout', 'Centipede', 'ChopperCommand', 'CrazyClimber', 
            'DemonAttack', 'DoubleDunk',  'Enduro', 'FishingDerby', 'Freeway', 
            'Frostbite', 'Gopher', 'Gravitar', 'IceHockey', 'Jamesbond',  
            'Kangaroo', 'Krull', 'KungFuMaster', 'MontezumaRevenge', 'MsPacman', 
            'NameThisGame', 'Pitfall', 'Pong', 'PrivateEye', 'Qbert', 
            'Riverraid', 'RoadRunner', 'Robotank', 'Seaquest', 'SpaceInvaders', 
            'StarGunner', 'Tennis', 'TimePilot', 'Tutankham', 'UpNDown', 
            'Venture', 'VideoPinball', 'WizardOfWor', 'Zaxxon', 
]

register_benchmark({
    'name' : 'Atari50_40M',
    'description' :'7 Atari games from Mnih et al. (2013), with pixel observations, 40M frames',
    'tasks'  : [{'env_id' : _game + _ATARI_SUFFIX, 'trials' : 3, 'num_timesteps' : int(40e6)} for _game in _atari50]
})
