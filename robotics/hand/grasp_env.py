#!/usr/bin/env python3
"""
Displays robot fetch at a disco party.
"""
from mujoco_py import load_model_from_path, MjSim, MjViewer
import math
import os

model = load_model_from_path("../assets/hand/grasp_block.xml")
sim = MjSim(model)

viewer = MjViewer(sim)

t = 0

while True:
    viewer.render()
    t += 1
    sim.step()
    state = sim.get_state()

    state.qpos[1] = 0.1*math.sin(0.01*t)
    state.qpos[0] = 0.1*math.cos(0.01*t)

    sim.set_state(state)
    # if t > 100 and os.getenv('TESTING') is not None:
    #     break
