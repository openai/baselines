# ACKTR

- Original paper: https://arxiv.org/abs/1708.05144
- Baselines blog post: https://blog.openai.com/baselines-acktr-a2c/
- `python -m baselines.run --alg=acktr --env=PongNoFrameskip-v4` runs the algorithm for 40M frames = 10M timesteps on an Atari Pong. See help (`-h`) for more options.
- also refer to the repo-wide [README.md](../../README.md#training-models)

## ACKTR with continuous action spaces
The code of ACKTR has been refactored to handle both discrete and continuous action spaces uniformly. In the original version, discrete and continuous action spaces were handled by different code (actkr_disc.py and acktr_cont.py) with little overlap. If interested in the original version of the acktr for continuous action spaces, use `old_acktr_cont` branch. Note that original code performs better on the mujoco tasks than the refactored version; we are still investigating why. 
