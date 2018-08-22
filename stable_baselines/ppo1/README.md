# PPO1

```
class stable_baselines.PPO1(policy, env, gamma=0.99, timesteps_per_actorbatch=256, clip_param=0.2, 
entcoeff=0.01, optim_epochs=4, optim_stepsize=1e-3, optim_batchsize=64, lam=0.95, adam_epsilon=1e-5, 
schedule='linear', verbose=0)
```

### Notes 

- Original paper: https://arxiv.org/abs/1707.06347
- Baselines blog post: https://blog.openai.com/openai-baselines-ppo/
- `mpirun -np 8 python -m baselines.ppo1.run_atari` runs the algorithm for 40M frames = 10M timesteps on an Atari game. See help (`-h`) for more options.
- `python -m baselines.ppo1.run_mujoco` runs the algorithm for 1M frames on a Mujoco environment.

- Train mujoco 3d humanoid (with optimal-ish hyperparameters): `mpirun -np 16 python -m baselines.ppo1.run_humanoid --model-path=/path/to/model`
- Render the 3d humanoid: `python -m baselines.ppo1.run_humanoid --play --model-path=/path/to/model`

### Can use
- Reccurent policies: :heavy_check_mark:
- Multi processing: :heavy_check_mark: (with MPI only)
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
|                 | **policy:** (function (str, Gym Spaces, Gym Spaces): TensorFlow Tensor) <br>&nbsp;&nbsp;&nbsp; creates the policy <br><br> **env:** (Gym environment or str) <br>&nbsp;&nbsp;&nbsp; The environment to learn from (if registered in Gym, can be str) <br><br> **timesteps_per_actorbatch:** (int) (default=256) <br>&nbsp;&nbsp;&nbsp; timesteps per actor per update <br><br> **clip_param:** (float) (default=0.2) <br>&nbsp;&nbsp;&nbsp; clipping parameter epsilon <br><br> **entcoeff:** (float) (default=0.01) <br>&nbsp;&nbsp;&nbsp; the entropy loss weight <br><br> **optim_epochs:** (float) (default=4) <br>&nbsp;&nbsp;&nbsp; the optimizer's number of epochs <br><br> **optim_stepsize:** (float) (default=1e-3) <br>&nbsp;&nbsp;&nbsp; the optimizer's stepsize <br><br> **optim_batchsize:** (int) (default=64) <br>&nbsp;&nbsp;&nbsp; the optimizer's the batch size <br><br> **gamma:** (float) (default=0.99) <br>&nbsp;&nbsp;&nbsp; discount factor <br><br> **lam:** (float) (default=0.95) <br>&nbsp;&nbsp;&nbsp; advantage estimation <br><br> **adam_epsilon:** (float) (default=1e-5) <br>&nbsp;&nbsp;&nbsp; the epsilon value for the adam optimizer <br><br> **schedule:** (str) (default='linear') <br>&nbsp;&nbsp;&nbsp; The type of scheduler for the learning rate update ('linear', 'constant', 'double_linear_con', 'middle_drop' or 'double_middle_drop') <br><br> **verbose:** (int) (default=0) <br>&nbsp;&nbsp;&nbsp; the verbosity level: 0 none, 1 training information, 2 tensorflow debug |
| **Attributes:** |     |
|                 | **env:** (Gym environment) <br>&nbsp;&nbsp;&nbsp; The environment to learn from |

### Example
```python
import gym

from stable_baselines.common.policies import MlpPolicy, MlpLstmPolicy, MlpLnLstmPolicy, \
    CnnPolicy, CnnLstmPolicy, CnnLnLstmPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import PPO1

env = gym.make('CartPole-v1')
env = DummyVecEnv([lambda: env])

model = PPO1(MlpPolicy, env, verbose=1)
model.learn(total_timesteps=25000)
model.save("ppo1_cartpole")

del model # remove to demonstrate saving and loading

PPO1.load("ppo1_cartpole")

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