import os

from gym.wrappers.monitoring import video_recorder

from stable_baselines import logger
from stable_baselines.common.vec_env import VecEnvWrapper, DummyVecEnv, VecNormalize, VecFrameStack, SubprocVecEnv


class VecVideoRecorder(VecEnvWrapper):
    """
    Wraps a VecEnv or VecEnvWrapper object to record rendered image as mp4 video.
    It requires ffmpeg or avconv to be installed on the machine.

    :param venv: (VecEnv or VecEnvWrapper)
    :param video_folder: (str) Where to save videos
    :param record_video_trigger: (func) Function that defines when to start recording.
                                        The function takes the current number of step,
                                        and returns whether we should start recording or not.
    :param video_length: (int)  Length of recorded videos
    :param name_prefix: (str) Prefix to the video name
    """

    def __init__(self, venv, video_folder, record_video_trigger,
                 video_length=200, name_prefix='rl-video'):

        VecEnvWrapper.__init__(self, venv)

        self.env = venv
        # Temp variable to retrieve metadata
        temp_env = venv

        # Unwrap to retrieve metadata dict
        # that will be used by gym recorder
        while isinstance(temp_env, VecNormalize) or isinstance(temp_env, VecFrameStack):
            temp_env = temp_env.venv

        if isinstance(temp_env, DummyVecEnv) or isinstance(temp_env, SubprocVecEnv):
            metadata = temp_env.get_attr('metadata')[0]
        else:
            metadata = temp_env.metadata

        self.env.metadata = metadata

        self.record_video_trigger = record_video_trigger
        self.video_recorder = None

        self.video_folder = os.path.abspath(video_folder)
        # Create output folder if needed
        os.makedirs(self.video_folder, exist_ok=True)

        self.name_prefix = name_prefix
        self.step_id = 0
        self.video_length = video_length

        self.recording = False
        self.recorded_frames = 0

    def reset(self):
        obs = self.venv.reset()
        self.start_video_recorder()
        return obs

    def start_video_recorder(self):
        self.close_video_recorder()

        video_name = '{}-step-{}-to-step-{}'.format(self.name_prefix, self.step_id,
                                                    self.step_id + self.video_length)
        base_path = os.path.join(self.video_folder, video_name)
        self.video_recorder = video_recorder.VideoRecorder(
                env=self.env,
                base_path=base_path,
                metadata={'step_id': self.step_id}
                )

        self.video_recorder.capture_frame()
        self.recorded_frames = 1
        self.recording = True

    def _video_enabled(self):
        return self.record_video_trigger(self.step_id)

    def step_wait(self):
        obs, rews, dones, infos = self.venv.step_wait()

        self.step_id += 1
        if self.recording:
            self.video_recorder.capture_frame()
            self.recorded_frames += 1
            if self.recorded_frames > self.video_length:
                logger.info("Saving video to ", self.video_recorder.path)
                self.close_video_recorder()
        elif self._video_enabled():
            self.start_video_recorder()

        return obs, rews, dones, infos

    def close_video_recorder(self):
        if self.recording:
            self.video_recorder.close()
        self.recording = False
        self.recorded_frames = 1

    def close(self):
        VecEnvWrapper.close(self)
        self.close_video_recorder()

    def __del__(self):
        self.close()
