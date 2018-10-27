import os
from baselines import logger
from baselines.common.vec_env import VecEnvWrapper
from gym.wrappers.monitoring import video_recorder


class VecVideoRecorder(VecEnvWrapper):
    def __init__(self, venv, directory, video_callable, video_length=200):
        VecEnvWrapper.__init__(self, venv)
        self.video_callable = video_callable
        self.video_recorder = None

        self.directory = os.path.abspath(directory)
        if not os.path.exists(self.directory): os.mkdir(self.directory)

        self.file_prefix = "vecenv"
        self.file_infix = '{}'.format(os.getpid())
        self.step_id = 0
        self.video_length = video_length

        self.recording = False
        self.recorded_frames = 0

    def reset(self):
        obs = self.venv.reset()

        self.start_video_recorder()

        return obs

    def start_video_recorder(self):
        self.close()

        base_path = os.path.join(self.directory, '{}.video.{}.video{:06}'.format(self.file_prefix, self.file_infix, self.step_id))
        self.video_recorder = video_recorder.VideoRecorder(
                env=self.venv,
                base_path=base_path,
                metadata={'step_id': self.step_id}
                )

        self.video_recorder.capture_frame()
        self.recorded_frames = 1
        self.recording = True

    def _video_enabled(self):
        return self.video_callable(self.step_id)

    def step_wait(self):
        obs, rews, dones, infos = self.venv.step_wait()

        self.step_id += 1
        if self.recording:
            self.video_recorder.capture_frame()
            self.recorded_frames += 1
            if self.recorded_frames > self.video_length:
                logger.info("Saving video to ", self.video_recorder.path)
                self.close()
        elif self._video_enabled():
                self.start_video_recorder()

        return obs, rews, dones, infos

    def close(self):
        if self.recording:
            self.video_recorder.close()
        self.recording = False
        self.recorded_frames = 0

    def __del__(self):
        self.close()
