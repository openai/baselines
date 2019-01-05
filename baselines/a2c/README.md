# A2C

- Original paper: https://arxiv.org/abs/1602.01783
- Baselines blog post: https://blog.openai.com/baselines-acktr-a2c/
- `python -m baselines.run --alg=a2c --env=PongNoFrameskip-v4` runs the algorithm for 40M frames = 10M timesteps on an Atari Pong. See help (`-h`) for more options
- also refer to the repo-wide [README.md](../../README.md#training-models)

## Files
- `run_atari`: file used to run the algorithm.
- `policies.py`: contains the different versions of the A2C architecture (MlpPolicy, CNNPolicy, LstmPolicy...).
- `a2c.py`: - Model : class used to initialize the step_model (sampling) and train_model (training)
	- learn : Main entrypoint for A2C algorithm. Train a policy with given network architecture on a given environment using a2c algorithm.
- `runner.py`: class used to generates a batch of experiences
