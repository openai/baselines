# DDPG

```
class stable_baselines.DDPG(policy, env, gamma=0.99, memory_policy=None, eval_env=None, nb_train_steps=50, 
nb_rollout_steps=100, nb_eval_steps=100, param_noise=None, action_noise=None, action_range=(-1., 1.), 
normalize_observations=False, tau=0.001, batch_size=128, param_noise_adaption_interval=50, 
normalize_returns=False, enable_popart=False, observation_range=(-5., 5.), critic_l2_reg=0., 
return_range=(-np.inf, np.inf), actor_lr=1e-4, critic_lr=1e-3, clip_norm=None, reward_scale=1., 
render=False, render_eval=False, layer_norm=True, memory_limit=100, verbose=0, _init_setup_model=True)
```

### Notes 

- Original paper: https://arxiv.org/abs/1509.02971
- Baselines post: https://blog.openai.com/better-exploration-with-parameter-noise/
- `python -m baselines.ddpg.main` runs the algorithm for 1M frames = 10M timesteps on a Mujoco environment. See help (`-h`) for more options.

### Can use
- Reccurent policies: :x:
- Multi processing: :x:
- Gym spaces:

| **space**     | **Action**         | **Observation**    |
| ------------- | ------------------ | ------------------ |
| Discrete      | :x:                | :heavy_check_mark: |
| Box           | :heavy_check_mark: | :heavy_check_mark: |
| MultiDiscrete | :x:                | :heavy_check_mark: |
| MultiBinary   | :x:                | :heavy_check_mark: |

### Parameters

| **Parameters:** |     |
| --------------- | --- |
|                 | **policy:** (ActorCriticPolicy) <br>&nbsp;&nbsp;&nbsp; the policy <br><br> **env:** (Gym environment or str) <br>&nbsp;&nbsp;&nbsp; The environment to learn from (if registered in Gym, can be str) <br><br> **gamma:** (float) (default=0.99) <br>&nbsp;&nbsp;&nbsp; the discount rate <br><br> **memory_policy:** (Memory) (default=None) <br>&nbsp;&nbsp;&nbsp; the replay buffer (if None, default to baselines.ddpg.memory.Memory) <br><br> **eval_env:** (Gym Environment) (default=None) <br>&nbsp;&nbsp;&nbsp; the evaluation environment (can be None) <br><br> **nb_train_steps:** (int) (default=50) <br>&nbsp;&nbsp;&nbsp; the number of training steps <br><br> **nb_rollout_steps:** (int) (default=100) <br>&nbsp;&nbsp;&nbsp; the number of rollout steps <br><br> **nb_eval_steps:** (int) (default=100) <br>&nbsp;&nbsp;&nbsp; the number of evalutation steps <br><br> **param_noise:** (AdaptiveParamNoiseSpec) (default=None) <br>&nbsp;&nbsp;&nbsp; the parameter noise type (can be None) <br><br> **action_noise:** (ActionNoise) (default=None) <br>&nbsp;&nbsp;&nbsp; the action noise type (can be None) <br><br> **param_noise_adaption_interval:** (int) (default=50) <br>&nbsp;&nbsp;&nbsp; apply param noise every N steps <br><br> **tau:** (float) (default=0.001) <br>&nbsp;&nbsp;&nbsp; the soft update coefficient (keep old values, between 0 and 1) <br><br> **normalize_returns:** (bool) (default=False) <br>&nbsp;&nbsp;&nbsp; should the critic output be normalized <br><br> **enable_popart:** (bool) (default=False) <br>&nbsp;&nbsp;&nbsp; enable pop-art normalization of the critic output (https://arxiv.org/pdf/1602.07714.pdf) <br><br> **normalize_observations:** (bool) (default=False) <br>&nbsp;&nbsp;&nbsp; should the observation be normalized <br><br> **batch_size:** (int) (default=128) <br>&nbsp;&nbsp;&nbsp; the size of the batch for learning the policy <br><br> **observation_range:** (tuple) (default=(-5., 5.)) <br>&nbsp;&nbsp;&nbsp; the bounding values for the observation <br><br> **action_range:** (tuple) (default=(-1., 1.)) <br>&nbsp;&nbsp;&nbsp; the bounding values for the actions <br><br> **return_range:** (tuple) (default=(-np.inf, np.inf)) <br>&nbsp;&nbsp;&nbsp; the bounding values for the critic output <br><br> **critic_l2_reg:** (float) (default=0.) <br>&nbsp;&nbsp;&nbsp; l2 regularizer coefficient <br><br> **actor_lr:** (float) (default=1e-4) <br>&nbsp;&nbsp;&nbsp; the actor learning rate <br><br> **critic_lr:** (float) (default=1e-3) <br>&nbsp;&nbsp;&nbsp; the critic learning rate <br><br> **clip_norm:** (float) (default=None) <br>&nbsp;&nbsp;&nbsp; clip the gradients (disabled if None) <br><br> **reward_scale:** (float) (default=1.) <br>&nbsp;&nbsp;&nbsp; the value the reward should be scaled by <br><br> **render:** (bool) (default=False) <br>&nbsp;&nbsp;&nbsp; enable rendering of the environment <br><br> **render_eval:** (bool) (default=False) <br>&nbsp;&nbsp;&nbsp; enable rendering of the evalution environment <br><br> **layer_norm:** (bool) (default=True) <br>&nbsp;&nbsp;&nbsp; enable layer normalization for the policies <br><br> **memory_limit:** (int) (default=100) <br>&nbsp;&nbsp;&nbsp; the max number of transitions to store <br><br> **verbose:** (int) (default=0) <br>&nbsp;&nbsp;&nbsp; the verbosity level: 0 none, 1 training information, 2 tensorflow debug |
| **Attributes:** |     |
|                 | **env:** (Gym environment) <br>&nbsp;&nbsp;&nbsp; The environment to learn from |

### Example
```python
import gym

from stable_baselines.common.policies import MlpPolicy, CnnPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines.ddpg.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise, AdaptiveParamNoiseSpec
from stable_baselines import DDPG

env = gym.make('CartPole-v1')
env = DummyVecEnv([lambda: env])

# the noise objects for DDPG
param_noise = None
action_noise = NormalActionNoise(mean=1, sigma=0)

model = DDPG(MlpPolicy, env, verbose=1, param_noise=param_noise, action_noise=action_noise)
model.learn(total_timesteps=25000)
model.save("ddpg_cartpole")

del model # remove to demonstrate saving and loading

DDPG.load("ddpg_cartpole")

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