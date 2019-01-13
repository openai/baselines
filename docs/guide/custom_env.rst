.. _custom_env:

Using Custom Environments
==========================

To use the rl baselines with custom environments, they just need to follow the *gym* interface.
That is to say, your environment must implement the following methods (and inherits from OpenAI Gym Class):


.. code-block:: python

  import gym
  from gym import spaces

  class CustomEnv(gym.Env):
    """Custom Environment that follows gym interface"""
    metadata = {'render.modes': ['human']}

    def __init__(self, arg1, arg2, ...):
      super(CustomEnv, self).__init__()
      # Define action and observation space
      # They must be gym.spaces objects
      # Example when using discrete actions:
      self.action_space = spaces.Discrete(N_DISCRETE_ACTIONS)
      # Example for using image as input:
      self.observation_space = spaces.Box(low=0, high=255,
                                          shape=(HEIGHT, WIDTH, N_CHANNELS), dtype=np.uint8)

    def step(self, action):
      ...
    def reset(self):
      ...
    def render(self, mode='human', close=False):
      ...


Then you can define and train a RL agent with:

.. code-block:: python

  # Instantiate and wrap the env
  env = DummyVecEnv([lambda: CustomEnv(arg1, ...)])
  # Define and Train the agent
  model = A2C(CnnPolicy, env).learn(total_timesteps=1000)


You can find a `complete guide online <https://github.com/openai/gym/tree/master/gym/envs#how-to-create-new-environments-for-gym>`_
on creating a custom Gym environment.


Optionally, you can also register the environment with gym,
that will allow you to create the RL agent in one line (and use ``gym.make()`` to instantiate the env).


In the project, for testing purposes, we use a custom environment named ``IdentityEnv``
defined `in this file <https://github.com/hill-a/stable-baselines/blob/master/stable_baselines/common/identity_env.py>`_.
An example of how to use it can be found `here <https://github.com/hill-a/stable-baselines/blob/master/tests/test_identity.py>`_.
