# trpo_mpi

- Original paper: https://arxiv.org/abs/1502.05477
- Baselines blog post https://blog.openai.com/openai-baselines-ppo/
- `mpirun -np 16 python -m baselines.ppo1.run_atari` runs the algorithm for 40M frames = 10M timesteps on an Atari game. See help (`-h`) for more options.
- `python -m baselines.ppo1.run_mujoco` runs the algorithm for 1M timesteps on a Mujoco environment.