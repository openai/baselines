# Generative Adversarial Imitation Learning (GAIL)

- Original paper: https://arxiv.org/abs/1606.03476

For results benchmarking on MuJoCo, please navigate to [here](result/gail-result.md)

## If you want to train an imitation learning agent

### Step 1: Download expert data

Download the expert data into `./data`, [download link](https://drive.google.com/drive/folders/1h3H4AY_ZBx08hz-Ct0Nxxus-V1melu1U?usp=sharing)

### Step 2: Run GAIL

Run with single thread:

```bash
python -m baselines.gail.run_mujoco
```

Run with multiple threads:

```bash
mpirun -np 16 python -m baselines.gail.run_mujoco
```

See help (`-h`) for more options.

#### In case you want to run Behavior Cloning (BC)

```bash
python -m baselines.gail.behavior_clone
```

See help (`-h`) for more options.

## Others

Thanks to the open source:

- @openai/imitation
- @carpedm20/deep-rl-tensorflow

Also, thanks [Ryan Julian](https://github.com/ryanjulian) for reviewing the code
