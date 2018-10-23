# explicitly import sub-packages to register algorithms

import baselines.a2c.a2c
import baselines.acer.acer
import baselines.acktr.acktr
import baselines.deepq.deepq
import baselines.ddpg.ddpg
import baselines.ppo2.ppo2

# not really sure why flake8 complains only about trpo_mpi here...
import baselines.trpo_mpi.trpo_mpi # noqa: F401

