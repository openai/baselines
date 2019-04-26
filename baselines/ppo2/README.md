# PPO2

- Original paper: https://arxiv.org/abs/1707.06347
- Baselines blog post: https://blog.openai.com/openai-baselines-ppo/

## Examples 
- `python -m baselines.run --alg=ppo2 --env=PongNoFrameskip-v4` runs the algorithm for 40M frames = 10M timesteps on an Atari Pong. See help (`-h`) for more options.
- `python -m baselines.run --alg=ppo2 --env=Ant-v2 --num_timesteps=1e6` runs the algorithm for 1M frames on a Mujoco Ant environment.

### RNN networks
- `python -m baselines.run --alg=ppo2 --env=PongNoFrameskip-v4 --network=ppo_cnn_lstm` runs on an Atari Pong with 
    `ppo_cnn_lstm` network.
- `python -m baselines.run --alg=ppo2 --env=Ant-v2 --num_timesteps=1e6 --network=ppo_lstm --value_network=copy` 
    runs on a Mujoco Ant environment with `ppo_lstm` network whose value and policy networks are separated, but have 
    same structure.

## See Also
- refer to the repo-wide [README.md](../../README.md#training-models)
