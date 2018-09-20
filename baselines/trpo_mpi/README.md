# trpo_mpi

- Original paper: https://arxiv.org/abs/1502.05477
- Baselines blog post https://blog.openai.com/openai-baselines-ppo/
- `mpirun -np 16 python -m baselines.run --alg=trpo_mpi --env=PongNoFrameskip-v4` runs the algorithm for 40M frames = 10M timesteps on an Atari Pong. See help (`-h`) for more options.
- `python -m baselines.run --alg=trpo_mpi --env=Ant-v2 --num_timesteps=1e6` runs the algorithm for 1M timesteps on a Mujoco Ant environment. 
- also refer to the repo-wide [README.md](../../README.md#training-models)
