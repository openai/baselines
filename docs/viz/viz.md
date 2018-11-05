# Loading and visualizing results 
In order to compare performance of algorithms, we often would like to visualize learning curves (reward as a function of time steps), or some other auxiliary information about learning
aggregated into a plot. Baselines repo provides tools for doing so in several different ways, depending on the goal. 

## Preliminaries
For all algorithms in baselines summary data is saved into a folder defined by logger. By default, a folder `$TMPDIR/openai-<date>-<time>` is used;
you can see the location of logger directory at the beginning of the training in the message like this:

```
Logging to /var/folders/mq/tgrn7bs17s1fnhlwt314b2fm0000gn/T/openai-2018-10-29-15-03-13-537078
```
The location can be changed by changing `OPENAI_LOGDIR` environment variable; for instance:
```bash
export OPENAI_LOGDIR=$HOME/logs/cartpole-ppo
python -m baselines.run --alg=ppo2 --env=CartPole-v0 --num_time steps=30000 --nsteps=128
```
will log data to `~/logs/cartpole-ppo`. 

## Using TensorBoard
One of the most straightforward ways to visualize data is to use [TensorBoard](https://www.tensorflow.org/guide/summaries_and_tensorboard). Baselines logger can dump data in tensorboard-compatible format; to 
set that up, set environment variables `OPENAI_LOG_FORMAT`
```bash
export OPENAI_LOG_FORMAT='stdout,log,csv,tensorboard' # formats are comma-separated, but for tensorboard you only really need the last one
```
And you can now start TensorBoard with:
```bash
tensorboard --logdir=$OPENAI_LOGDIR
```

## Loading summaries of the results
If the summary overview provided by tensorboard is not sufficient, and you would like to either access to raw environment episode data, or use complex post-processing notavailable in tensorboard, you can load results into python as [pandas](https://pandas.pydata.org/) dataframes. 
The colab notebook with the full version of the code is available [here](https://colab.research.google.com/drive/1Wez1SA9PmNkCoYc8Fvl53bhU3F8OffGm) (use "Open in playground" button to get a runnable version) 

For instance, the following snippet:
```python
from baselines.common import plot_util as pu
results = pu.load_results('~/logs/cartpole-ppo') 
```
will search for all folders with baselines-compatible results in `~/logs/cartpole-ppo` and subfolders and 
return a list of Result objects. Each Result object is a named tuple with the following fields:

- dirname: str - name of the folder from which data was loaded

- metadata: dict) - dictionary with various metadata (read from metadata.json file)

- progress: pandas.DataFrame - tabular data saved by logger as a pandas dataframe. Available if csv is in logger formats. 

- monitor: pandas.DataFrame - raw episode data (length, episode reward, timestamp). Available if environment wrapped with [Monitor](../../baselines/bench/monitor.py) wrapper

## Plotting: single- and few curve plots
Once results are loaded, they can be plotted in all conventional means. For example:
```python
import matplotlib.pyplot as plt
import numpy as np
r = results[0]
plt.plot(np.cumsum(r.monitor.l), r.monitor.r)
```
will print a (very noisy learning curve) for CartPole (assuming we ran the training command for CartPole above). Note the cumulative sum trick to get convert length of the episode into number of time steps taken so far.

<img src="https://storage.googleapis.com/baselines/assets/viz/Screen%20Shot%202018-10-29%20at%204.44.46%20PM.png" width="500">

We can get a smoothened version of the same curve by using `plot_util.smooth()` function:
```python
plt.plot(np.cumsum(r.monitor.l), pu.smooth(r.monitor.r, radius=10))
```

<img src="https://storage.googleapis.com/baselines/assets/viz/Screen%20Shot%202018-10-29%20at%204.49.13%20PM.png" width="730">

We can also get a similar curve by using logger summaries (instead of raw episode data in monitor.csv): 
```python
plt.plot(r.progress.total_time steps, r.progress.eprewmean)
```

<img src="https://storage.googleapis.com/baselines/assets/viz/Screen%20Shot%202018-10-29%20at%205.04.31%20PM.png" width="730">

Note, however, that raw episode data is stored by the Monitor wrapper, and hence looks similar for all algorithms, whereas progress data
is handled by the algorithm itself, and hence can vary (column names, type of data available) between algorithms. 

## Plotting: many curves 
While the loading and the plotting functions described above in principle give you access to any slice of the training summaries,
sometimes it is necessary to plot and compare many training runs (multiple algorithms, multiple seeds for random number generator),
and usage of the functions above can get tedious and messy. For that case, `baselines.common.plot_util` provides convenience function
`plot_results` that handles multiple Result objects that need to be routed in multiple plots. Consider the following bash snippet that
runs ppo2 with cartpole with 6 different seeds for 30k time steps, first with batch size 32, and then with batch size 128:

```bash
for seed in $(seq 0 5); do
OPENAI_LOGDIR=$HOME/logs/cartpole-ppo/b32-$seed python -m baselines.run --alg=ppo2 --env=CartPole-v0 --num_time steps=3e4 --seed=$seed --nsteps=32
done
for seed in $(seq 0 5); do
OPENAI_LOGDIR=$HOME/logs/cartpole-ppo/b128-$seed python -m baselines.run --alg=ppo2 --env=CartPole-v0 --num_time steps=3e4 --seed=$seed --nsteps=128
done
```
These 12 runs can be loaded just as before:
```python
results = pu.load_results('~/logs/cartpole-ppo')
```
But how do we plot all 12 of them in a sensible manner? `baselines.common.plot_util` module provides `plot_results` function to do just that:
```
results = results[1:]
pu.plot_results(results)
```
(note that now the length of the results list is 13, due to the data from the previous run stored directly in `~/logs/cartpole-ppo`; we discard first element for the same reason)
The results are split into two groups based on batch size and are plotted on a separate graph. More specifically, by default `plot_results` considers digits after dash at the end of the directory name to be seed id and groups the runs that differ only by those together. 

<img src="https://storage.googleapis.com/baselines/assets/viz/Screen%20Shot%202018-10-29%20at%205.53.45%20PM.png" width="700">

Showing all seeds on the same plot may be somewhat hard to comprehend and analyse. We can instead average over all seeds via the following command:
<img  src="https://storage.googleapis.com/baselines/assets/viz/Screen%20Shot%202018-11-02%20at%204.42.52%20PM.png" width="720">

The lighter shade shows the standard deviation of data, and darker shade - 
error in estimate of the mean (that is, standard deviation divided by square root of number of seeds)
Note that averaging over seeds requires resampling to a common grid, which, in turn, requires smoothing
(using language of signal processing, we need to do low-pass filtering before resampling to avoid aliasing effects). 
You can change the amount of smoothing by adjusting `resample` and `smooth_step` arguments to achieve desired smoothing effect
See the docstring of `plot_util` function for more info. 

