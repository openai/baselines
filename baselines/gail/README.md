# GAIL

- Original paper: https://arxiv.org/abs/1606.03476

## If you want to train an imitation learning agent

### Step 1: Download expert data

Download the expert data into `./data`

### Step 2: Imitation learning

```bash
python -m baselines.gail.run_mujoco
```

Run with multiple threads:

```bash
mpirun -np 16 python -m baselines.gail.run_mujoco
```

See help (`-h`) for more options.


