# PPOSGD

- Original paper: https://arxiv.org/abs/1707.06347
- Baselines blog post: https://blog.openai.com/openai-baselines-ppo/
- `mpirun -np 8 python -m baselines.ppo1.run_atari` runs the algorithm for 40M frames = 10M timesteps on an Atari game. See help (`-h`) for more options.
- `python -m baselines.ppo1.run_mujoco` runs the algorithm for 1M frames on a Mujoco environment.

