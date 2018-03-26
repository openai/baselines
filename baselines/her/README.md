# Hindsight Experience Replay
For details on Hindsight Experience Replay (HER), please read the [paper](https://arxiv.org/abs/1707.01495).

## How to use Hindsight Experience Replay

### Getting started
Training an agent is very simple:
```bash
python -m baselines.her.experiment.train
```
This will train a DDPG+HER agent on the `FetchReach` environment.
You should see the success rate go up quickly to `1.0`, which means that the agent achieves the
desired goal in 100% of the cases.
The training script logs other diagnostics as well and pickles the best policy so far (w.r.t. to its test success rate),
the latest policy, and, if enabled, a history of policies every K epochs.

To inspect what the agent has learned, use the play script:
```bash
python -m baselines.her.experiment.play /path/to/an/experiment/policy_best.pkl
```
You can try it right now with the results of the training step (the script prints out the path for you).
This should visualize the current policy for 10 episodes and will also print statistics.


### Reproducing results
In order to reproduce the results from [Plappert et al. (2018)](https://arxiv.org/abs/1802.09464), run the following command:
```bash
python -m baselines.her.experiment.train --num_cpu 19
```
This will require a machine with sufficient amount of physical CPU cores. In our experiments,
we used [Azure's D15v2 instances](https://docs.microsoft.com/en-us/azure/virtual-machines/linux/sizes),
which have 20 physical cores. We only scheduled the experiment on 19 of those to leave some head-room on the system.
