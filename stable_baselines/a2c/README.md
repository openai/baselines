# A2C

```
class stable_baselines.A2C(policy, env, gamma=0.99, n_steps=5, vf_coef=0.25, ent_coef=0.01, max_grad_norm=0.5,
learning_rate=7e-4, alpha=0.99, epsilon=1e-5, lr_schedule='linear', verbose=0, tensorboard_log=None)
```

### Notes 

- Original paper: https://arxiv.org/abs/1602.01783
- Baselines blog post: https://blog.openai.com/baselines-acktr-a2c/
- `python -m baselines.a2c.run_atari` runs the algorithm for 40M frames = 10M timesteps on an Atari game. See help (`-h`) for more options.

### Can use
- Reccurent policies: :heavy_check_mark:
- Multi processing: :heavy_check_mark:
- Gym spaces:

| **space**     | **Action**         | **Observation**    |
| ------------- | ------------------ | ------------------ |
| Discrete      | :heavy_check_mark: | :heavy_check_mark: |
| Box           | :heavy_check_mark: | :heavy_check_mark: |
| MultiDiscrete | :heavy_check_mark: | :heavy_check_mark: |
| MultiBinary   | :heavy_check_mark: | :heavy_check_mark: |

### Parameters

| **Parameters:** |     |
| --------------- | --- |
|                 | **policy:** (ActorCriticPolicy) <br>&nbsp;&nbsp;&nbsp; The policy model to use (MLP, CNN, LSTM, ...) <br><br> **env:** (Gym environment or str) <br>&nbsp;&nbsp;&nbsp; The environment to learn from (if registered in Gym, can be str) <br><br> **gamma:** (float) (default=0.99) <br>&nbsp;&nbsp;&nbsp; Discount factor <br><br> **n_steps:** (int) (default=5) <br>&nbsp;&nbsp;&nbsp; The number of steps to run for each environment <br><br> **vf_coef:** (float) (default=0.25) <br>&nbsp;&nbsp;&nbsp; Value function coefficient for the loss calculation <br><br> **ent_coef:** (float) (default=0.01) <br>&nbsp;&nbsp;&nbsp; Entropy coefficient for the loss caculation <br><br> **max_grad_norm:** (float) (default=0.5) <br>&nbsp;&nbsp;&nbsp; The maximum value for the gradient clipping <br><br> **learning_rate:** (float) (default=7e-4) <br>&nbsp;&nbsp;&nbsp; The learning rate <br><br> **alpha:** (float) (default=0.99) <br>&nbsp;&nbsp;&nbsp; RMS prop optimizer decay <br><br> **epsilon:** (float) (default=1e-5) <br>&nbsp;&nbsp;&nbsp; RMS prop optimizer epsilon <br><br> **lr_schedule:** (str) (default='linear') <br>&nbsp;&nbsp;&nbsp; The type of scheduler for the learning rate update ('linear', 'constant', 'double_linear_con', 'middle_drop' or 'double_middle_drop') <br><br> **verbose:** (int) (default=0) <br>&nbsp;&nbsp;&nbsp; the verbosity level: 0 none, 1 training information, 2 tensorflow debug <br><br> **tensorboard_log:** (str) (default=None) <br>&nbsp;&nbsp;&nbsp; the log location for tensorboard (if None, no logging) |
| **Attributes:** |     |
|                 | **env:** (Gym environment) <br>&nbsp;&nbsp;&nbsp; The environment to learn from <br><br> **graph:** (TensorFlow Graph) <br>&nbsp;&nbsp;&nbsp; The tensorflow graph of the model <br><br> **sess:** (TensorFlow Session) <br>&nbsp;&nbsp;&nbsp; The tensorflow session of the model <br><br> **params:** ([TensorFlow Variable]) <br>&nbsp;&nbsp;&nbsp; The tensorflow parameters of the model|

### Example
```python
import gym

from stable_baselines.common.policies import MlpPolicy, MlpLstmPolicy, MlpLnLstmPolicy, \
    CnnPolicy, CnnLstmPolicy, CnnLnLstmPolicy
from stable_baselines.common.vec_env import SubprocVecEnv
from stable_baselines import A2C

# multiprocess environment
n_cpu = 4
env = SubprocVecEnv([lambda: gym.make('CartPole-v1') for i in range(n_cpu)])

model = A2C(MlpPolicy, env, verbose=1)
model.learn(total_timesteps=25000)
model.save("a2c_cartpole")

del model # remove to demonstrate saving and loading

A2C.load("a2c_cartpole")

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