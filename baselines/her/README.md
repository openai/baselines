# Hindsight Experience Replay
For details on Hindsight Experience Replay (HER), please read the [paper](https://arxiv.org/pdf/1707.01495.pdf).

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


### Advanced usage
The train script comes with advanced features like MPI support, that allows to scale across all cores of a single machine.
To see all available options, simply run this command:
```bash
python -m baselines.her.experiment.train --help
```
To run on, say, 20 CPU cores, you can use the following command:
```bash
python -m baselines.her.experiment.train --num_cpu 20
```
That's it, you are now running rollouts using 20 MPI workers and average gradients for network updates across all 20 core.
