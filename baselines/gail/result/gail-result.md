# Results of GAIL/BC on Mujoco

Here's the extensive experimental results of applying GAIL/BC on Mujoco environments, including 
Hopper-v1, Walker2d-v1, HalfCheetah-v1, Humanoid-v1, HumanoidStandup-v1. Every imitator is evaluated with seed to be 0.

## Results

### Training through iterations

- Hoppers-v1
<img src='hopper-training.png'> 

- HalfCheetah-v1
<img src='halfcheetah-training.png'> 

- Walker2d-v1
<img src='walker2d-training.png'> 

- Humanoid-v1
<img src='humanoid-training.png'> 

- HumanoidStandup-v1
<img src='humanoidstandup-training.png'> 

For details (e.g., adversarial loss, discriminator accuracy, etc.) about GAIL training, please see [here](https://drive.google.com/drive/folders/1nnU8dqAV9i37-_5_vWIspyFUJFQLCsDD?usp=sharing)

### Determinstic Policy (Set std=0)
|   | Un-normalized | Normalized |
|---|---|---|
| Hopper-v1 | <img src='Hopper-unnormalized-deterministic-scores.png'> | <img src='Hopper-normalized-deterministic-scores.png'> |
| HalfCheetah-v1 | <img src='HalfCheetah-unnormalized-deterministic-scores.png'> | <img src='HalfCheetah-normalized-deterministic-scores.png'> |
| Walker2d-v1 | <img src='Walker2d-unnormalized-deterministic-scores.png'> | <img src='Walker2d-normalized-deterministic-scores.png'> |
| Humanoid-v1 | <img src='Humanoid-unnormalized-deterministic-scores.png'> | <img src='Humanoid-normalized-deterministic-scores.png'> |
| HumanoidStandup-v1 | <img src='HumanoidStandup-unnormalized-deterministic-scores.png'> | <img src='HumanoidStandup-normalized-deterministic-scores.png'> |

### Stochatic Policy 
|   | Un-normalized | Normalized |
|---|---|---|
| Hopper-v1 | <img src='Hopper-unnormalized-stochastic-scores.png'> | <img src='Hopper-normalized-stochastic-scores.png'> |
| HalfCheetah-v1 | <img src='HalfCheetah-unnormalized-stochastic-scores.png'> | <img src='HalfCheetah-normalized-stochastic-scores.png'> |
| Walker2d-v1 | <img src='Walker2d-unnormalized-stochastic-scores.png'> | <img src='Walker2d-normalized-stochastic-scores.png'> |
| Humanoid-v1 | <img src='Humanoid-unnormalized-stochastic-scores.png'> | <img src='Humanoid-normalized-stochastic-scores.png'> |
| HumanoidStandup-v1 | <img src='HumanoidStandup-unnormalized-stochastic-scores.png'> | <img src='HumanoidStandup-normalized-stochastic-scores.png'> |

### details about GAIL imitator

For all environments, the 
imitator is trained with 1, 5, 10, 50 trajectories, where each trajectory contains at most 
1024 transitions, and seed 0, 1, 2, 3, respectively.

### details about the BC imitators

All BC imitators are trained with seed 0.
