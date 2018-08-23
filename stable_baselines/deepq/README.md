# DeepQ

```
class stable_baselines.DeepQ(policy, env, gamma=0.99, learning_rate=5e-4, buffer_size=50000, 
exploration_fraction=0.1, exploration_final_eps=0.02, train_freq=1, batch_size=32, checkpoint_freq=10000, 
checkpoint_path=None, learning_starts=1000, target_network_update_freq=500, prioritized_replay=False,
prioritized_replay_alpha=0.6, prioritized_replay_beta0=0.4, prioritized_replay_beta_iters=None, 
prioritized_replay_eps=1e-6, param_noise=False, verbose=0, tensorboard_log=None)
```

### Notes 

- Original paper: https://arxiv.org/abs/1312.5602

### Can use
- Reccurent policies: :x:
- Multi processing: :x:
- Gym spaces:

| **space**     | **Action**         | **Observation**    |
| ------------- | ------------------ | ------------------ |
| Discrete      | :heavy_check_mark: | :heavy_check_mark: |
| Box           | :x:                | :heavy_check_mark: |
| MultiDiscrete | :x:                | :heavy_check_mark: |
| MultiBinary   | :x:                | :heavy_check_mark: |

### Parameters

| **Parameters:** |     |
| --------------- | --- |
|                 | **policy:** (function (TensorFlow Tensor, int, str, bool): TensorFlow Tensor) <br>&nbsp;&nbsp;&nbsp; the policy that takes the following inputs: <br>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; - observation_in: (object) the output of observation placeholder <br>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; - num_actions: (int) number of actions <br>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; - scope: (str) <br>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; - reuse: (bool) should be passed to outer variable scope <br>&nbsp;&nbsp;&nbsp; and returns a tensor of shape (batch_size, num_actions) with values of every action. <br><br> **env:** (Gym environment or str) <br>&nbsp;&nbsp;&nbsp; The environment to learn from (if registered in Gym, can be str) <br><br> **gamma:** (float) (default=0.99) <br>&nbsp;&nbsp;&nbsp; discount factor <br><br> **learning_rate:** (float) (default=5e-4) <br>&nbsp;&nbsp;&nbsp; learning rate for adam optimizer <br><br> **buffer_size:** (int) (default=50000) <br>&nbsp;&nbsp;&nbsp; size of the replay buffer <br><br> **exploration_fraction:** (float) (default=0.1) <br>&nbsp;&nbsp;&nbsp; fraction of entire training period over which the exploration rate is annealed <br><br> **exploration_final_eps:** (float) (default=0.02) <br>&nbsp;&nbsp;&nbsp; final value of random action probability <br><br> **train_freq:** (int) (default=1) <br>&nbsp;&nbsp;&nbsp; update the model every `train_freq` steps. set to None to disable printing <br><br> **batch_size:** (int) (default=32) <br>&nbsp;&nbsp;&nbsp; size of a batched sampled from replay buffer for training <br><br> **checkpoint_freq:** (int) (default=10000) <br>&nbsp;&nbsp;&nbsp; how often to save the model. This is so that the best version is restored at the end of the training. If you do not wish to restore the best version at the end of the training set this variable to None. <br><br> **checkpoint_path:** (str) (default=None) <br>&nbsp;&nbsp;&nbsp; replacement path used if you need to log to somewhere else than a temporary directory. <br><br> **learning_starts:** (int) (default=1000) <br>&nbsp;&nbsp;&nbsp; how many steps of the model to collect transitions for before learning starts <br><br> **target_network_update_freq:** (int) (default=500) <br>&nbsp;&nbsp;&nbsp; update the target network every `target_network_update_freq` steps. <br><br> **prioritized_replay:** (bool) (default=False) <br>&nbsp;&nbsp;&nbsp; if True prioritized replay buffer will be used. <br><br> **prioritized_replay_alpha:** (float) (default=0.6) <br>&nbsp;&nbsp;&nbsp; alpha parameter for prioritized replay buffer <br><br> **prioritized_replay_beta0:** (float) (default=0.4) <br>&nbsp;&nbsp;&nbsp; initial value of beta for prioritized replay buffer <br><br> **prioritized_replay_beta_iters:** (int) (default=None) <br>&nbsp;&nbsp;&nbsp; number of iterations over which beta will be annealed from initial value to 1.0. If set to None equals to max_timesteps. <br><br> **prioritized_replay_eps:** (float) (default=1e-6) <br>&nbsp;&nbsp;&nbsp; epsilon to add to the TD errors when updating priorities. <br><br> **param_noise:** (bool) (default=False) <br>&nbsp;&nbsp;&nbsp; Whether or not to apply noise to the parameters of the policy. <br><br> **verbose:** (int) (default=0) <br>&nbsp;&nbsp;&nbsp; the verbosity level: 0 none, 1 training information, 2 tensorflow debug <br><br> **tensorboard_log:** (str) (default=None) <br>&nbsp;&nbsp;&nbsp; the log location for tensorboard (if None, no logging) |
| **Attributes:** |     |
|                 | **env:** (Gym environment) <br>&nbsp;&nbsp;&nbsp; The environment to learn from <br><br> **graph:** (TensorFlow Graph) <br>&nbsp;&nbsp;&nbsp; The tensorflow graph of the model <br><br> **sess:** (TensorFlow Session) <br>&nbsp;&nbsp;&nbsp; The tensorflow session of the model <br><br> **params:** ([TensorFlow Variable]) <br>&nbsp;&nbsp;&nbsp; The tensorflow parameters of the model|

### Example
```python
import gym

from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines.deepq.models import mlp, cnn_to_mlp
from stable_baselines import DeepQ

env = gym.make('CartPole-v1')
env = DummyVecEnv([lambda: env])

model = DeepQ(mlp(hiddens=[32]), env, verbose=1)
model.learn(total_timesteps=25000)
model.save("deepq_cartpole")

del model # remove to demonstrate saving and loading

DeepQ.load("deepq_cartpole")

obs = env.reset()
while True:
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    env.render()
```

and with atari

```python
from stable_baselines.common.atari_wrappers import make_atari
from stable_baselines.deepq.models import mlp, cnn_to_mlp
from stable_baselines import DeepQ

env = make_atari('BreakoutNoFrameskip-v4')

# nature CNN for DeepQ
cnn_policy = cnn_to_mlp(
	convs=[(32, 8, 4), (64, 4, 2), (64, 3, 1)],
    hiddens=[256],
    dueling=True)

model = DeepQ(cnn_policy, env, verbose=1)
model.learn(total_timesteps=25000)
model.save("deepq_breakout")

del model # remove to demonstrate saving and loading

DeepQ.load("deepq_breakout")

obs = env.reset()
while True:
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    env.render()
```

### Methods 
|                                                              |                                                                                                        |
| ------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------ |
| **get_env** ()                                               | returns the current environment (can be None if not defined)                                           |
| **set_env** (env)                                            | Checks the validity of the environment, and if it is coherent, set it as the current environment.      |
| **learn** (total_timesteps [, callback, seed, log_interval]) | Train the model on the environment, and returns the trained model.                                     |
| **predict** (observation [, state, mask])                    | Get the model's action from an observation                                                             |
| **action_probability** (observation [, state, mask])         | Get the model's action probability distribution from an observation                                    |
| **save** (save_path)                                         | Save the current parameters to file                                                                    |
| **load** (load_path [, env, **kwargs])                       | Load the model from file                                                                               |