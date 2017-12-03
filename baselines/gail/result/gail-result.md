# Results for GAIL on Mujoco

Here's the extensive results for applying GAIL on Mujoco environment, including 
Hopper, Walker2d, HalfCheetah, Humanoid, HumanoidStandup. For all environments, the 
imitator is trained with 1, 5, 10, 50 trajectories, where each trajectory contains at most 
1024 transitions, and seed 0, 1, 2, 3, respectively.

## Results

|   | Un-normalized | Normalized |
|---|---|---|
| Hopper-v1 | <img src='Hopper-unnormalized-scores.png'> | <img src='Hopper-normalized-scores.png'> |
| HalfCheetah-v1 | <img src='HalfCheetah-unnormalized-scores.png'> | <img src='HalfCheetah-normalized-scores.png'> |
| Walker2d-v1 | <img src='Walker2d-unnormalized-scores.png'> | <img src='Walker2d-normalized-scores.png'> |
| Humanoid-v1 | <img src='Humanoid-unnormalized-scores.png'> | <img src='Humanoid-normalized-scores.png'> |
| HumanoidStandup-v1 | <img src='HumanoidStandup-unnormalized-scores.png'> | <img src='HumanoidStandup-normalized-scores.png'> |

### details
Each imitator is evaluated with random seed equals to 0.
