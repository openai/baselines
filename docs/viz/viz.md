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
```
export OPENAI_LOGDIR=$HOME/models/mujoco-ppo-humanoid
python -m baselines.run --alg=ppo2 --env=Humanoid-v2
```
will log data to `~/models/mujoco-ppo-humanoid`. 

## Using TensorBoard
One of the most straightforward ways to visualize data is to use TensorBoard (). Baselines logger can dump data in tensorboard-compatible format.

## Loading summaries of the results

## Plotting: standalone 

## Plotting: jupyter notebook

