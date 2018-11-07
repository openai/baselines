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


## Hindsight Experience Replay with Demonstrations
Using pre-recorded demonstrations to Overcome the exploration problem in HER based Reinforcement learning.
For details, please read the [paper](https://arxiv.org/pdf/1709.10089.pdf).

### Getting started
The first step is to generate the demonstration dataset. This can be done in two ways, either by using a VR system to manipulate the arm using physical VR trackers or the simpler way is to write a script to carry out the respective task. Now some tasks can be complex and thus it would be difficult to write a hardcoded script for that task (eg. Fetch Push), but here our focus is on providing an algorithm that helps the agent to learn from demonstrations, and not on the demonstration generation paradigm itself. Thus the data collection part is left to the reader's choice.

We provide a script for the Fetch Pick and Place task, to generate demonstrations for the Pick and Place task execute:
```bash
python experiment/data_generation/fetch_data_generation.py
```
This outputs ```data_fetch_random_100.npz``` file which is our data file.

#### Configuration
The provided configuration is for training an agent with HER without demonstrations, we need to change a few paramters for the HER algorithm to learn through demonstrations, to do that, set:

* bc_loss: 1 - whether or not to use the behavior cloning loss as an auxilliary loss
* q_filter: 1 - whether or not a Q value filter should be used on the Actor outputs
* num_demo: 100 - number of expert demo episodes
* demo_batch_size: 128 - number of samples to be used from the demonstrations buffer, per mpi thread
* prm_loss_weight: 0.001 - Weight corresponding to the primary loss
* aux_loss_weight:  0.0078 - Weight corresponding to the auxilliary loss also called the cloning loss

Apart from these changes the reported results also have the following configurational changes:

* n_cycles: 20 - per epoch
* batch_size: 1024 - per mpi thread, total batch size
* random_eps: 0.1  - percentage of time a random action is taken
* noise_eps: 0.1  - std of gaussian noise added to not-completely-random actions

Now training an agent with pre-recorded demonstrations:
```bash
python -m baselines.her.experiment.train --env=FetchPickAndPlace-v0 --n_epochs=1000 --demo_file=/Path/to/demo_file.npz --num_cpu=1
```

This will train a DDPG+HER agent on the `FetchPickAndPlace` environment by using previously generated demonstration data.
To inspect what the agent has learned, use the play script as described above.

### Results
Training with demonstrations helps overcome the exploration problem and achieves a faster and better convergence. The following graphs contrast the difference between training with and without demonstration data, We report the mean Q values vs Epoch and the Success Rate vs Epoch:


<div class="imgcap" align="middle">
<center><img src="../../data/fetchPickAndPlaceContrast.png"></center>
<div class="thecap" align="middle"><b>Training results for Fetch Pick and Place task constrasting between training with and without demonstration data.</b></div>
</div>
