# A2C

- Original paper: https://arxiv.org/abs/1602.01783
- Baselines blog post: https://blog.openai.com/baselines-acktr-a2c/
- `python -m baselines.a2c.run_atari` runs the algorithm for 40M frames = 10M timesteps on an Atari game. See help (`-h`) for more options.

## Files
- `run_atari`: file used to run the algorithm.
- `policies.py`: contains the different versions of the A2C architecture (MlpPolicy, CNNPolicy, LstmPolicy...).
- `a2c.py`: - Model : class used to initialize the step_model (sampling) and train_model (training)
	- Runner: class used to generates a batch of experiences


