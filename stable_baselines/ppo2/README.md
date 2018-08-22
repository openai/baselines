# PPO2

```
class stable_baselines.PPO2(policy, env, gamma=0.99, n_steps=128, ent_coef=0.01, learning_rate=2.5e-4, 
vf_coef=0.5, max_grad_norm=0.5, lam=0.95, nminibatches=4, noptepochs=4, cliprange=0.2, verbose=0)
```

### Notes 

- Original paper: https://arxiv.org/abs/1707.06347
- Baselines blog post: https://blog.openai.com/openai-baselines-ppo/
- `python -m baselines.ppo2.run_atari` runs the algorithm for 40M frames = 10M timesteps on an Atari game. See help (`-h`) for more options.
- `python -m baselines.ppo2.run_mujoco` runs the algorithm for 1M frames on a Mujoco environment.

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
|                 | **policy:** (ActorCriticPolicy) <br>&nbsp;&nbsp;&nbsp; The policy model to use (MLP, CNN, LSTM, ...) <br><br> **env:** (Gym environment or str) <br>&nbsp;&nbsp;&nbsp; The environment to learn from (if registered in Gym, can be str) <br><br> **gamma:** (float) (default=0.99) <br>&nbsp;&nbsp;&nbsp; Discount factor <br><br> **n_steps:** (int) (default=128) <br>&nbsp;&nbsp;&nbsp; The number of steps to run for each environment <br><br> **ent_coef:** (float) (default=0.01) <br>&nbsp;&nbsp;&nbsp; Entropy coefficient for the loss caculation <br><br> **learning_rate:** (float or callable) (default=2.5e-4) <br>&nbsp;&nbsp;&nbsp; The learning rate, it can be a function <br><br> **vf_coef:** (float) (default=0.5) <br>&nbsp;&nbsp;&nbsp; Value function coefficient for the loss calculation <br><br> **max_grad_norm:** (float) (default=0.5) <br>&nbsp;&nbsp;&nbsp; The maximum value for the gradient clipping <br><br> **lam:** (float) (default=0.95) <br>&nbsp;&nbsp;&nbsp; Factor for trade-off of bias vs variance for Generalized Advantage Estimator <br><br> **nminibatches:** (int) (default=4) <br>&nbsp;&nbsp;&nbsp; Number of minibatches for the policies <br><br> **noptepochs:** (int) (default=4) <br>&nbsp;&nbsp;&nbsp; Number of epoch when optimizing the surrogate <br><br> **cliprange:** (float or callable) (default=0.2) <br>&nbsp;&nbsp;&nbsp; Clipping parameter, it can be a function <br><br> **verbose:** (int) (default=0) <br>&nbsp;&nbsp;&nbsp; the verbosity level: 0 none, 1 training information, 2 tensorflow debug|
| **Attributes:** |     |
|                 | **env:** (Gym environment) <br>&nbsp;&nbsp;&nbsp; The environment to learn from |

### Example
```python
import gym

from stable_baselines.common.policies import MlpPolicy, MlpLstmPolicy, MlpLnLstmPolicy, \
    CnnPolicy, CnnLstmPolicy, CnnLnLstmPolicy
from stable_baselines.common.vec_env import SubprocVecEnv
from stable_baselines import PPO2

# multiprocess environment
n_cpu = 4
env = SubprocVecEnv([lambda: gym.make('CartPole-v1') for i in range(n_cpu)])

model = PPO2(MlpPolicy, env, verbose=1)
model.learn(total_timesteps=25000)
model.save("ppo2_cartpole")

del model # remove to demonstrate saving and loading

PPO2.load("ppo2_cartpole")

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