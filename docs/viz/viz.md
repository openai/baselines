# Loading and visualizing results 
In order to compare performance of algorithms, we often would like to vizualise learning curves (reward as a function of timesteps), or some other auxiliary information about learining
aggregated into a plot. Baselines repo provides tools for doing so in several different ways, depending on the goal. 

## Preliminaries
For all algorithms in baselines directory in which summary data is saved is defined by logger. By default, a folder  `$TMPDIR/openai-<date>-<time>` is used; 
you can see the location of logger directory at the beginning of the training in the message like this:

```
Logging to /var/folders/mq/tgrn7bs17s1fnhlwt314b2fm0000gn/T/openai-2018-10-29-15-03-13-537078
```
The location can be changed by changing `OPENAI_LOGDIR` environment variable; for instance:
```bash
export OPENAI_LOGDIR=$HOME/models/mujoco-ppo-humanoid
python -m baselines.run --alg=ppo2 --env=Humanoid-v2
```
will log data to `~/models/mujoco-ppo-humanoid`. 

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
For instance, the following snippet:
```python
from baselines.common import plot_util
results = plot_util.load_results('<path_to_logdir>') 
```
will search for all folders with baselines-compatible results in `<path_to_logdir>` and subfolders and 
return a list of Result objects. Each Result object is a named tuple with the following fields:

- dirname: str - name of the folder from which data was loaded

- metadata: dict) - dictionary with various metadata (read from metadata.json file)

- progress: pandas.DataFrame - tabular data saved by logger as a pandas dataframe. Available if csv is in logger formats. 

- monitor: pandas.DataFrame - raw episode data (length, episode reward, timestamp). Available if environment wrapped with [Monitor](baselines/bench/monitor.py) wrapper


## Plotting: standalone 

## Plotting: jupyter notebook

