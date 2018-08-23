# ACKTR

```
class stable_baselines.ACKTR(policy, env, gamma=0.99, nprocs=1, n_steps=20, ent_coef=0.01, vf_coef=0.25, 
vf_fisher_coef=1.0, learning_rate=0.25, max_grad_norm=0.5, kfac_clip=0.001, lr_schedule='linear', verbose=0,
tensorboard_log=None)
```

### Notes 

- Original paper: https://arxiv.org/abs/1708.05144
- Baselines blog post: https://blog.openai.com/baselines-acktr-a2c/
- `python -m baselines.acktr.run_atari` runs the algorithm for 40M frames = 10M timesteps on an Atari game. See help (`-h`) for more options.

### Can use
- Reccurent policies: :heavy_check_mark:
- Multi processing: :heavy_check_mark:
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
|                 | **policy:** (ActorCriticPolicy) <br>&nbsp;&nbsp;&nbsp; The policy model to use (MLP, CNN, LSTM, ...) <br><br> **env:** (Gym environment or str) <br>&nbsp;&nbsp;&nbsp; The environment to learn from (if registered in Gym, can be str) <br><br> **gamma:** (float) (default=0.99) <br>&nbsp;&nbsp;&nbsp; Discount factor <br><br> **nprocs:** (int) (default=1) <br>&nbsp;&nbsp;&nbsp; The number of threads for TensorFlow operations <br><br> **n_steps:** (int) (default=20) <br>&nbsp;&nbsp;&nbsp; The number of steps to run for each environment <br><br> **ent_coef:** (float) (default=0.01) <br>&nbsp;&nbsp;&nbsp; The weight for the entropic loss <br><br> **vf_coef:** (float) (default=0.25) <br>&nbsp;&nbsp;&nbsp; The weight for the loss on the value function <br><br> **vf_fisher_coef:** (float) (default=1.0) <br>&nbsp;&nbsp;&nbsp; The weight for the fisher loss on the value function <br><br> **learning_rate:** (float) (default=0.25) <br>&nbsp;&nbsp;&nbsp; The initial learning rate for the RMS prop optimizer <br><br> **max_grad_norm:** (float) (default=0.5) <br>&nbsp;&nbsp;&nbsp; The clipping value for the maximum gradient <br><br> **kfac_clip:** (float) (default=0.001) <br>&nbsp;&nbsp;&nbsp; gradient clipping for Kullback leiber <br><br> **lr_schedule:** (str) (default='linear') <br>&nbsp;&nbsp;&nbsp; The type of scheduler for the learning rate update ('linear', 'constant', 'double_linear_con', 'middle_drop' or 'double_middle_drop') <br><br> **verbose:** (int) (default=0) <br>&nbsp;&nbsp;&nbsp; the verbosity level: 0 none, 1 training information, 2 tensorflow debug <br><br> **tensorboard_log:** (str) (default=None) <br>&nbsp;&nbsp;&nbsp; the log location for tensorboard (if None, no logging) |
| **Attributes:** |     |
|                 | **env:** (Gym environment) <br>&nbsp;&nbsp;&nbsp; The environment to learn from <br><br> **graph:** (TensorFlow Graph) <br>&nbsp;&nbsp;&nbsp; The tensorflow graph of the model <br><br> **sess:** (TensorFlow Session) <br>&nbsp;&nbsp;&nbsp; The tensorflow session of the model <br><br> **params:** ([TensorFlow Variable]) <br>&nbsp;&nbsp;&nbsp; The tensorflow parameters of the model|

### Example
```python
import gym

from stable_baselines.common.policies import MlpPolicy, MlpLstmPolicy, MlpLnLstmPolicy, \
    CnnPolicy, CnnLstmPolicy, CnnLnLstmPolicy
from stable_baselines.common.vec_env import SubprocVecEnv
from stable_baselines import ACKTR

# multiprocess environment
n_cpu = 4
env = SubprocVecEnv([lambda: gym.make('CartPole-v1') for i in range(n_cpu)])

model = ACKTR(MlpPolicy, env, verbose=1)
model.learn(total_timesteps=25000)
model.save("acktr_cartpole")

del model # remove to demonstrate saving and loading

ACKTR.load("acktr_cartpole")

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