from collections import deque
import cv2
cv2.ocl.setUseOpenCL(False)
from .atari_wrappers import WarpFrame, ClipRewardEnv, FrameStack, ScaledFloatFrame
from .wrappers import TimeLimit
import numpy as np
import gym


class StochasticFrameSkip(gym.Wrapper):
    def __init__(self, env, n, stickprob):
        gym.Wrapper.__init__(self, env)
        self.n = n
        self.stickprob = stickprob
        self.curac = None
        self.rng = np.random.RandomState()
        self.supports_want_render = hasattr(env, "supports_want_render")

    def reset(self, **kwargs):
        self.curac = None
        return self.env.reset(**kwargs)

    def step(self, ac):
        done = False
        totrew = 0
        for i in range(self.n):
            # First step after reset, use action
            if self.curac is None:
                self.curac = ac
            # First substep, delay with probability=stickprob
            elif i==0:
                if self.rng.rand() > self.stickprob:
                    self.curac = ac
            # Second substep, new action definitely kicks in
            elif i==1:
                self.curac = ac
            if self.supports_want_render and i<self.n-1:
                ob, rew, done, info = self.env.step(self.curac, want_render=False)
            else:
                ob, rew, done, info = self.env.step(self.curac)
            totrew += rew
            if done: break
        return ob, totrew, done, info

    def seed(self, s):
        self.rng.seed(s)

class PartialFrameStack(gym.Wrapper):
    def __init__(self, env, k, channel=1):
        """
        Stack one channel (channel keyword) from previous frames
        """
        gym.Wrapper.__init__(self, env)
        shp = env.observation_space.shape
        self.channel = channel
        self.observation_space = gym.spaces.Box(low=0, high=255,
            shape=(shp[0], shp[1], shp[2] + k - 1),
            dtype=env.observation_space.dtype)
        self.k = k
        self.frames = deque([], maxlen=k)
        shp = env.observation_space.shape

    def reset(self):
        ob = self.env.reset()
        assert ob.shape[2] > self.channel
        for _ in range(self.k):
            self.frames.append(ob)
        return self._get_ob()

    def step(self, ac):
        ob, reward, done, info = self.env.step(ac)
        self.frames.append(ob)
        return self._get_ob(), reward, done, info

    def _get_ob(self):
        assert len(self.frames) == self.k
        return np.concatenate([frame if i==self.k-1 else frame[:,:,self.channel:self.channel+1]
            for (i, frame) in enumerate(self.frames)], axis=2)

class Downsample(gym.ObservationWrapper):
    def __init__(self, env, ratio):
        """
        Downsample images by a factor of ratio
        """
        gym.ObservationWrapper.__init__(self, env)
        (oldh, oldw, oldc) = env.observation_space.shape
        newshape = (oldh//ratio, oldw//ratio, oldc)
        self.observation_space = gym.spaces.Box(low=0, high=255,
            shape=newshape, dtype=np.uint8)

    def observation(self, frame):
        height, width, _ = self.observation_space.shape
        frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
        if frame.ndim == 2:
            frame = frame[:,:,None]
        return frame

class Rgb2gray(gym.ObservationWrapper):
    def __init__(self, env):
        """
        Downsample images by a factor of ratio
        """
        gym.ObservationWrapper.__init__(self, env)
        (oldh, oldw, _oldc) = env.observation_space.shape
        self.observation_space = gym.spaces.Box(low=0, high=255,
            shape=(oldh, oldw, 1), dtype=np.uint8)

    def observation(self, frame):
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        return frame[:,:,None]


class MovieRecord(gym.Wrapper):
    def __init__(self, env, savedir, k):
        gym.Wrapper.__init__(self, env)
        self.savedir = savedir
        self.k = k
        self.epcount = 0
    def reset(self):
        if self.epcount % self.k == 0:
            self.env.unwrapped.movie_path = self.savedir
        else:
            self.env.unwrapped.movie_path = None
            self.env.unwrapped.movie = None
        self.epcount += 1
        return self.env.reset()

class AppendTimeout(gym.Wrapper):
    def __init__(self, env):
        gym.Wrapper.__init__(self, env)
        self.action_space = env.action_space
        self.timeout_space = gym.spaces.Box(low=np.array([0.0]), high=np.array([1.0]), dtype=np.float32)
        self.original_os = env.observation_space
        if isinstance(self.original_os, gym.spaces.Dict):
            import copy
            ordered_dict = copy.deepcopy(self.original_os.spaces)
            ordered_dict['value_estimation_timeout'] = self.timeout_space
            self.observation_space = gym.spaces.Dict(ordered_dict)
            self.dict_mode = True
        else:
            self.observation_space = gym.spaces.Dict({
                'original': self.original_os,
                'value_estimation_timeout': self.timeout_space
                })
            self.dict_mode = False
        self.ac_count = None
        while 1:
            if not hasattr(env, "_max_episode_steps"):  # Looking for TimeLimit wrapper that has this field
                env = env.env
                continue
            break
        self.timeout = env._max_episode_steps

    def step(self, ac):
        self.ac_count += 1
        ob, rew, done, info = self.env.step(ac)
        return self._process(ob), rew, done, info

    def reset(self):
        self.ac_count = 0
        return self._process(self.env.reset())

    def _process(self, ob):
        fracmissing = 1 - self.ac_count / self.timeout
        if self.dict_mode:
            ob['value_estimation_timeout'] = fracmissing
        else:
            return { 'original': ob, 'value_estimation_timeout': fracmissing }

class StartDoingRandomActionsWrapper(gym.Wrapper):
    """
    Warning: can eat info dicts, not good if you depend on them
    """
    def __init__(self, env, max_random_steps, on_startup=True, every_episode=False):
        gym.Wrapper.__init__(self, env)
        self.on_startup = on_startup
        self.every_episode = every_episode
        self.random_steps = max_random_steps
        self.last_obs = None
        if on_startup:
            self.some_random_steps()

    def some_random_steps(self):
        self.last_obs = self.env.reset()
        n = np.random.randint(self.random_steps)
        #print("running for random %i frames" % n)
        for _ in range(n):
            self.last_obs, _, done, _ = self.env.step(self.env.action_space.sample())
            if done: self.last_obs = self.env.reset()

    def reset(self):
        return self.last_obs

    def step(self, a):
        self.last_obs, rew, done, info = self.env.step(a)
        if done:
            self.last_obs = self.env.reset()
            if self.every_episode:
                self.some_random_steps()
        return self.last_obs, rew, done, info

def make_retro(*, game, state=None, max_episode_steps=4500, **kwargs):
    import retro
    if state is None:
        state = retro.State.DEFAULT
    env = retro.make(game, state, **kwargs)
    env = StochasticFrameSkip(env, n=4, stickprob=0.25)
    if max_episode_steps is not None:
        env = TimeLimit(env, max_episode_steps=max_episode_steps)
    return env

def wrap_deepmind_retro(env, scale=True, frame_stack=4):
    """
    Configure environment for retro games, using config similar to DeepMind-style Atari in wrap_deepmind
    """
    env = WarpFrame(env)
    env = ClipRewardEnv(env)
    if frame_stack > 1:
        env = FrameStack(env, frame_stack)
    if scale:
        env = ScaledFloatFrame(env)
    return env

class SonicDiscretizer(gym.ActionWrapper):
    """
    Wrap a gym-retro environment and make it use discrete
    actions for the Sonic game.
    """
    def __init__(self, env):
        super(SonicDiscretizer, self).__init__(env)
        buttons = ["B", "A", "MODE", "START", "UP", "DOWN", "LEFT", "RIGHT", "C", "Y", "X", "Z"]
        actions = [['LEFT'], ['RIGHT'], ['LEFT', 'DOWN'], ['RIGHT', 'DOWN'], ['DOWN'],
                   ['DOWN', 'B'], ['B']]
        self._actions = []
        for action in actions:
            arr = np.array([False] * 12)
            for button in action:
                arr[buttons.index(button)] = True
            self._actions.append(arr)
        self.action_space = gym.spaces.Discrete(len(self._actions))

    def action(self, a): # pylint: disable=W0221
        return self._actions[a].copy()

class RewardScaler(gym.RewardWrapper):
    """
    Bring rewards to a reasonable scale for PPO.
    This is incredibly important and effects performance
    drastically.
    """
    def __init__(self, env, scale=0.01):
        super(RewardScaler, self).__init__(env)
        self.scale = scale

    def reward(self, reward):
        return reward * self.scale

class AllowBacktracking(gym.Wrapper):
    """
    Use deltas in max(X) as the reward, rather than deltas
    in X. This way, agents are not discouraged too heavily
    from exploring backwards if there is no way to advance
    head-on in the level.
    """
    def __init__(self, env):
        super(AllowBacktracking, self).__init__(env)
        self._cur_x = 0
        self._max_x = 0

    def reset(self, **kwargs): # pylint: disable=E0202
        self._cur_x = 0
        self._max_x = 0
        return self.env.reset(**kwargs)

    def step(self, action): # pylint: disable=E0202
        obs, rew, done, info = self.env.step(action)
        self._cur_x += rew
        rew = max(0, self._cur_x - self._max_x)
        self._max_x = max(self._max_x, self._cur_x)
        return obs, rew, done, info
