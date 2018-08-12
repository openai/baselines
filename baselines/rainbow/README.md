## This is a tensorflow implementation of Rainbow: Combining Improvements in Deep Reinforcement Learning from Google DeepMind based on OpenAI baselines.

## If you wish to experiment with the algorithm

##### Check out the examples

- [baselines/rainbow/experiments/atari/rainbow.py](experiments/atari/train.py) - more robust setup for training at scale.
- By default, dueling dqn is disabled, since it affect the stability of rainbow for the moment(Not observed in DeepMind 
rainbow paper.).
You can run the code as follows.
```bash
python -m baselines.rainbow.experiments.atari.rainbow --env Pong --save-dir Pong 
```

- Various improvements can be enabled by using different argument options.
- Standard C51 can be obtained just disable other options such as dueling, double and so on.

## Note
- Noisy network is used based on openai paper(Parameter Space Noise for Exploration) and stabilized. 
- Dueling seems incompatible when all the other improvements are enabled.
- N steps is not included.
## Reference
1. Human-level control through deep reinforcement learning(https://www.nature.com/nature/journal/v518/n7540/abs/nature14236.html)
2. Deep Reinforcement Learning with Double Q-Learning.(https://arxiv.org/abs/1509.06461) 
3. Prioritized Experience Replay(https://arxiv.org/abs/1511.05952)
4. Dueling Network Architectures for Deep Reinforcement Learning(https://arxiv.org/abs/1511.06581) 
5. Parameter Space Noise for Exploration(https://arxiv.org/abs/1706.01905)
5. A Distributional Perspective on Reinforcement Learning (https://arxiv.org/abs/1707.06887)
6. Rainbow: Combining Improvements in Deep Reinforcement Learning (https://arxiv.org/abs/1710.02298)
7. openai baselines(https://github.com/openai/baselines.git)
8. coach (https://github.com/NervanaSystems/coach.git)