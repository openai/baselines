# A2C

- Original paper: https://arxiv.org/abs/1602.01783
- Baselines blog post: https://blog.openai.com/baselines-acktr-a2c/

To run the algorithm on an Atari game (Breakout by default) for 40M frames = 10M timesteps:

```shell
python -m baselines.a2c.train_atari
```

Load the saved model and look at the trained policy:

```shell
python -m baselines.a2c.enjoy_atari
```

See help (`-h`) for more options.
