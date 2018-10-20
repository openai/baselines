from baselines.common.vec_env import VecEnvWrapper
import time
import os
import gym
from gym import Wrapper
from gym.wrappers.monitoring import stats_recorder, video_recorder


class VecVideoRecorder(VecEnvWrapper):
    def __init__(self, venv, directory, video_callable=None):
        VecEnvWrapper.__init__(self, venv)
        self.video_callable = video_callable
        self.video_recorder = None

        self.directory = os.path.abspath(directory)
        self.file_prefix = "vecenv"
        self.file_infix = '{}'.format(os.getpid())
        self.episode_id = 0

    def reset(self):
        obs = self.venv.reset()

        self.start_video_recorder()

        return obs

    def start_video_recorder(self):
        if self.video_recorder:
            print("Out")
            self.video_recorder.close()

        if self.venv.metadata is None:
            self.venv.metadata = {'render.modes': ['rgb_array']}
        self.video_recorder = video_recorder.VideoRecorder(
                env=self.venv,
                base_path=os.path.join(self.directory, '{}.video.{}.video{:06}'.format(self.file_prefix, self.file_infix, self.episode_id)),
                metadata={'episode_id': self.episode_id},
                enabled=self._video_enabled(),
                )
        self.video_recorder.capture_frame()

    def _video_enabled(self):
        return self.video_callable(self.episode_id)

    def step_wait(self):
        obs, rews, dones, infos = self.venv.step_wait()
        self.video_recorder.capture_frame()
        self.episode_id += 1
        if self.episode_id % 200 == 0:
            self.start_video_recorder()

        return obs, rews, dones, infos
