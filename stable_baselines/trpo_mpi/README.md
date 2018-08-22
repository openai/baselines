# TRPO

```
class stable_baselines.TRPO(policy, env, gamma=0.99, timesteps_per_batch=1024, max_kl=0.01, cg_iters=10, lam=0.98,
entcoeff=0.0, cg_damping=1e-2, vf_stepsize=3e-4, vf_iters=3, verbose=0)
```

### Notes 

- Original paper: https://arxiv.org/abs/1502.05477
- Baselines blog post https://blog.openai.com/openai-baselines-ppo/
- `mpirun -np 16 python -m baselines.trpo_mpi.run_atari` runs the algorithm for 40M frames = 10M timesteps on an Atari game. See help (`-h`) for more options.
- `python -m baselines.trpo_mpi.run_mujoco` runs the algorithm for 1M timesteps on a Mujoco environment.

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
|                 | **policy:** (function (str, Gym Space, Gym Space, bool): MLPPolicy) <br>&nbsp;&nbsp;&nbsp; policy generator
**env:** (Gym environment or str) <br>&nbsp;&nbsp;&nbsp; The environment to learn from (if registered in Gym, can be str)
**gamma:** (float) (default=0.99) <br>&nbsp;&nbsp;&nbsp; the discount value
**timesteps_per_batch:** (int) (default=1024) <br>&nbsp;&nbsp;&nbsp; the number of timesteps to run per batch (horizon)
**max_kl:** (float) (default=0.01) <br>&nbsp;&nbsp;&nbsp; the kullback leiber loss threshold
**cg_iters:** (int) (default=10) <br>&nbsp;&nbsp;&nbsp; the number of iterations for the conjugate gradient calculation
**lam:** (float) (default=0.98) <br>&nbsp;&nbsp;&nbsp; GAE factor
**entcoeff:** (float) (default=0.0) <br>&nbsp;&nbsp;&nbsp; the weight for the entropy loss
**cg_damping:** (float) (default=1e-2) <br>&nbsp;&nbsp;&nbsp; the compute gradient dampening factor
**vf_stepsize:** (float) (default=3e-4) <br>&nbsp;&nbsp;&nbsp; the value function stepsize
**vf_iters:** (int) (default=3) <br>&nbsp;&nbsp;&nbsp; the value function's number iterations for learning
**verbose:** (int) (default=0) <br>&nbsp;&nbsp;&nbsp; the verbosity level: 0 none, 1 training information, 2 tensorflow debug |
| **Attributes:** |     |
|                 | **env:** (Gym environment) <br>&nbsp;&nbsp;&nbsp; The environment to learn from |

### Example
```python
import gym

from stable_baselines.common.policies import MlpPolicy, MlpLstmPolicy, MlpLnLstmPolicy, \
    CnnPolicy, CnnLstmPolicy, CnnLnLstmPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import TRPO

env = gym.make('CartPole-v1')
env = DummyVecEnv([lambda: env])

model = TRPO(MlpPolicy, env, verbose=1)
model.learn(total_timesteps=25000)
model.save("trpo_cartpole")

del model # remove to demonstrate saving and loading

TRPO.load("trpo_cartpole")

obs = env.reset()
while True:
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    env.render()
```

###Methods 
|                                                              |                                                                                                        |
| ------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------ |
| **get_env** ()                                               | returns the current environment (can be None if not defined)                                           |
| **set_env** (env)                                            | Checks the validity of the environment, and if it is coherent, set it as the current environment.      |
| **learn** (total_timesteps [, callback, seed, log_interval]) | Train the model on the environment, and returns the trained model.                                     |
| **predict** (observation [, state, mask])                    | Get the model's action from an observation                                                             |
| **action_probability** (observation [, state, mask])         | Get the model's action probability distribution from an observation                                    |
| **save** (save_path)                                         | Save the current parameters to file                                                                    |
| **load** (load_path [, env, **kwargs])                       | Load the model from file                                                                               |