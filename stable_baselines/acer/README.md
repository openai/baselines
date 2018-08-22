# ACER

```
class stable_baselines.ACER(policy, env, gamma=0.99, n_steps=20, num_procs=1, q_coef=0.5, ent_coef=0.01, 
max_grad_norm=10, learning_rate=7e-4, lr_schedule='linear', rprop_alpha=0.99, rprop_epsilon=1e-5, 
buffer_size=5000, replay_ratio=4, replay_start=1000, correction_term=10.0, trust_region=True, alpha=0.99, 
delta=1, verbose=0)
```

### Notes 

- Original paper: https://arxiv.org/abs/1611.01224
- `python -m baselines.acer.run_atari` runs the algorithm for 40M frames = 10M timesteps on an Atari game. See help (`-h`) for more options.

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
|                 | **policy:** (ActorCriticPolicy) <br>&nbsp;&nbsp;&nbsp; The policy model to use (MLP, CNN, LSTM, ...)   <br><br> **env:** (Gym environment or str) <br>&nbsp;&nbsp;&nbsp; The environment to learn from (if registered in Gym, can be str)   <br><br> **gamma:** (float) (default=0.99) <br>&nbsp;&nbsp;&nbsp; The discount value  <br><br> **n_steps:** (int) (default=20) <br>&nbsp;&nbsp;&nbsp; The number of steps to run for each environment  <br><br> **num_procs:** (int) (default=1) <br>&nbsp;&nbsp;&nbsp; The number of threads for TensorFlow operations  <br><br> **q_coef:** (float) (default=0.5) <br>&nbsp;&nbsp;&nbsp; The weight for the loss on the Q value  <br><br> **ent_coef:** (float) (default=0.01) <br>&nbsp;&nbsp;&nbsp; The weight for the entropic loss  <br><br> **max_grad_norm:** (float) (default=10) <br>&nbsp;&nbsp;&nbsp; The clipping value for the maximum gradient  <br><br> **learning_rate:** (float) (default=7e-4) <br>&nbsp;&nbsp;&nbsp; The initial learning rate for the RMS prop optimizer  <br><br> **lr_schedule:** (str) (default='linear') <br>&nbsp;&nbsp;&nbsp; The type of scheduler for the learning rate update ('linear', 'constant', 'double_linear_con', 'middle_drop' or 'double_middle_drop')  <br><br> **rprop_epsilon:** (float) (default=1e-5) <br>&nbsp;&nbsp;&nbsp; RMS prop optimizer epsilon  <br><br> **rprop_alpha:** (float) (default=0.99) <br>&nbsp;&nbsp;&nbsp; RMS prop optimizer decay  <br><br> **buffer_size:** (int) (default=5000) <br>&nbsp;&nbsp;&nbsp; The buffer size in number of steps  <br><br> **replay_ratio:** (float) (default=4) <br>&nbsp;&nbsp;&nbsp; The number of replay learning per on policy learning on average, using a poisson distribution  <br><br> **replay_start:** (int) (default=1000) <br>&nbsp;&nbsp;&nbsp; The minimum number of steps in the buffer, before learning replay  <br><br> **correction_term:** (float) (default=10.0) <br>&nbsp;&nbsp;&nbsp; The correction term for the weights  <br><br> **trust_region:** (bool) (default=True) <br>&nbsp;&nbsp;&nbsp; Enable Trust region policy optimization loss  <br><br> **alpha:** (float) (default=0.99) <br>&nbsp;&nbsp;&nbsp; The decay rate for the Exponential moving average of the parameters  <br><br> **delta:** (float) (default=1) <br>&nbsp;&nbsp;&nbsp; trust region delta value  <br><br> **verbose:** (int) (default=0) <br>&nbsp;&nbsp;&nbsp; the verbosity level: 0 none, 1 training information, 2 tensorflow debug |
| **Attributes:** |     |
|                 | **env:** (Gym environment) <br>&nbsp;&nbsp;&nbsp; The environment to learn from |

### Example
```python
import gym

from stable_baselines.common.policies import MlpPolicy, MlpLstmPolicy, MlpLnLstmPolicy, \
    CnnPolicy, CnnLstmPolicy, CnnLnLstmPolicy
from stable_baselines.common.vec_env import SubprocVecEnv
from stable_baselines import ACER

# multiprocess environment
n_cpu = 4
env = SubprocVecEnv([lambda: gym.make('CartPole-v1') for i in range(n_cpu)])

model = ACER(MlpPolicy, env, verbose=1)
model.learn(total_timesteps=25000)
model.save("acer_cartpole")

del model # remove to demonstrate saving and loading

ACER.load("acer_cartpole")

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