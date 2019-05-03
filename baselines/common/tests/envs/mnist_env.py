import os.path as osp
import numpy as np
import tempfile
from gym import Env
from gym.spaces import Discrete, Box



class MnistEnv(Env):
    def __init__(
            self,
            episode_len=None,
            no_images=None
    ):
        import filelock
        from tensorflow.examples.tutorials.mnist import input_data
        # we could use temporary directory for this with a context manager and
        # TemporaryDirecotry, but then each test that uses mnist would re-download the data
        # this way the data is not cleaned up, but we only download it once per machine
        mnist_path = osp.join(tempfile.gettempdir(), 'MNIST_data')
        with filelock.FileLock(mnist_path + '.lock'):
           self.mnist = input_data.read_data_sets(mnist_path)

        self.np_random = np.random.RandomState()

        self.observation_space = Box(low=0.0, high=1.0, shape=(28,28,1))
        self.action_space = Discrete(10)
        self.episode_len = episode_len
        self.time = 0
        self.no_images = no_images

        self.train_mode()
        self.reset()

    def reset(self):
        self._choose_next_state()
        self.time = 0

        return self.state[0]

    def step(self, actions):
        rew = self._get_reward(actions)
        self._choose_next_state()
        done = False
        if self.episode_len and self.time >= self.episode_len:
            rew = 0
            done = True

        return self.state[0], rew, done, {}

    def seed(self, seed=None):
        self.np_random.seed(seed)

    def train_mode(self):
        self.dataset = self.mnist.train

    def test_mode(self):
        self.dataset = self.mnist.test

    def _choose_next_state(self):
        max_index = (self.no_images if self.no_images is not None else self.dataset.num_examples) - 1
        index = self.np_random.randint(0, max_index)
        image = self.dataset.images[index].reshape(28,28,1)*255
        label = self.dataset.labels[index]
        self.state = (image, label)
        self.time += 1

    def _get_reward(self, actions):
        return 1 if self.state[1] == actions else 0


