## If you are curious.

##### Train a Cartpole agent and watch it play once it converges!

Here's a list of commands to run to quickly get a working example:

<img src="../../data/cartpole.gif" width="25%" />


```bash
# Train model and save the results to cartpole_model.pkl
python -m baselines.deepq.experiments.train_cartpole
# Load the model saved in cartpole_model.pkl and visualize the learned policy
python -m baselines.deepq.experiments.enjoy_cartpole
```


Be sure to check out the source code of [both](experiments/train_cartpole.py) [files](experiments/enjoy_cartpole.py)!

## If you wish to apply DQN to solve a problem.

Check out our simple agent trained with one stop shop `deepq.learn` function. 

- [baselines/deepq/experiments/train_cartpole.py](experiments/train_cartpole.py) - train a Cartpole agent.
- [baselines/deepq/experiments/train_pong.py](experiments/train_pong.py) - train a Pong agent using convolutional neural networks.

In particular notice that once `deepq.learn` finishes training it returns `act` function which can be used to select actions in the environment. Once trained you can easily save it and load at later time. For both of the files listed above there are complimentary files `enjoy_cartpole.py` and `enjoy_pong.py` respectively, that load and visualize the learned policy.

## If you wish to experiment with the algorithm

##### Check out the examples


- [baselines/deepq/experiments/custom_cartpole.py](experiments/custom_cartpole.py) - Cartpole training with more fine grained control over the internals of DQN algorithm.
- [baselines/deepq/experiments/atari/train.py](experiments/atari/train.py) - more robust setup for training at scale.


##### Download a pretrained Atari agent

For some research projects it is sometimes useful to have an already trained agent handy. There's a variety of models to choose from. You can list them all by running:

```bash
python -m baselines.deepq.experiments.atari.download_model
```

Once you pick a model, you can download it and visualize the learned policy. Be sure to pass `--dueling` flag to visualization script when using dueling models.

```bash
python -m baselines.deepq.experiments.atari.download_model --blob model-atari-duel-pong-1 --model-dir /tmp/models
python -m baselines.deepq.experiments.atari.enjoy --model-dir /tmp/models/model-atari-duel-pong-1 --env Pong --dueling

```
