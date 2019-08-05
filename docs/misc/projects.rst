.. _projects:

Projects
=========

This is a list of projects using stable-baselines.
Please tell us, if you want your project to appear on this page ;)


Learning to drive in a day
--------------------------

Implementation of reinforcement learning approach to make a donkey car learn to drive.
Uses DDPG on VAE features (reproducing paper from wayve.ai)

| Author: Roma Sokolkov (@r7vme)
| Github repo: https://github.com/r7vme/learning-to-drive-in-a-day


Donkey Gym
----------

OpenAI gym environment for donkeycar simulator.

| Author: Tawn Kramer (@tawnkramer)
| Github repo: https://github.com/tawnkramer/donkey_gym


Self-driving FZERO Artificial Intelligence
------------------------------------------

Series of videos on how to make a self-driving FZERO artificial intelligence using reinforcement learning algorithms PPO2 and A2C.

| Author: Lucas Thompson
| `Video Link <https://www.youtube.com/watch?v=PT9pQliUXDk&list=PLTWFMbPFsvz2LIR7thpuU738FcRQbR_8I>`_


S-RL Toolbox
------------

S-RL Toolbox: Reinforcement Learning (RL) and State Representation Learning (SRL) for Robotics.
Stable-Baselines was originally developped for this project.

| Authors: Antonin Raffin, Ashley Hill, René Traoré, Timothée Lesort, Natalia Díaz-Rodríguez, David Filliat
| Github repo: https://github.com/araffin/robotics-rl-srl


Roboschool simulations training on Amazon SageMaker
---------------------------------------------------

"In this notebook example, we will make HalfCheetah learn to walk using the stable-baselines [...]"


| Author: Amazon AWS
| `Repo Link <https://github.com/awslabs/amazon-sagemaker-examples/tree/master/reinforcement_learning/rl_roboschool_stable_baselines>`_


MarathonEnvs + OpenAi.Baselines
-------------------------------


Experimental - using OpenAI baselines with MarathonEnvs (ML-Agents)


| Author: Joe Booth (@Sohojoe)
| Github repo: https://github.com/Sohojoe/MarathonEnvsBaselines


Learning to drive smoothly in minutes
-------------------------------------

Implementation of reinforcement learning approach to make a car learn to drive smoothly in minutes.
Uses SAC on VAE features.

| Author: Antonin Raffin (@araffin)
| Blog post: https://towardsdatascience.com/learning-to-drive-smoothly-in-minutes-450a7cdb35f4
| Github repo: https://github.com/araffin/learning-to-drive-in-5-minutes


Making Roboy move with elegance
-------------------------------

Project around Roboy, a tendon-driven robot, that enabled it to move its shoulder in simulation to reach a pre-defined point in 3D space. The agent used Proximal Policy Optimization (PPO) or Soft Actor-Critic (SAC) and was tested on the real hardware.

| Authors: Alexander Pakakis, Baris Yazici, Tomas Ruiz
| Email: FirstName.LastName@tum.de
| GitHub repo: https://github.com/Roboy/DeepAndReinforced
| DockerHub image: deepandreinforced/rl:latest
| Presentation: https://tinyurl.com/DeepRoboyControl
| Video: https://tinyurl.com/DeepRoboyControlVideo
| Blog post: https://tinyurl.com/mediumDRC
| Website: https://roboy.org/

Train a ROS-integrated mobile robot (differential drive) to avoid dynamic objects
---------------------------------------------------------------------------------

The RL-agent serves as local planner and is trained in a simulator, fusion of the Flatland Simulator and the crowd simulator Pedsim. This was tested on a real mobile robot.
The Proximal Policy Optimization (PPO) algorithm is applied.

| Author: Ronja Güldenring
| Email: 6guelden@informatik.uni-hamburg.de
| Video: https://www.youtube.com/watch?v=laGrLaMaeT4
| GitHub: https://github.com/RGring/drl_local_planner_ros_stable_baselines

Adversarial Policies: Attacking Deep Reinforcement Learning
-----------------------------------------------------------

Uses Stable Baselines to train *adversarial policies* that attack pre-trained victim policies in a zero-sum multi-agent environments.
May be useful as an example of how to integrate Stable Baselines with `Ray <https://github.com/ray-project/ray>`_ to perform distributed experiments and `Sacred <https://github.com/IDSIA/sacred>`_ for experiment configuration and monitoring.

| Authors: Adam Gleave, Michael Dennis, Neel Kant, Cody Wild
| Email: adam@gleave.me
| GitHub: https://github.com/HumanCompatibleAI/adversarial-policies
| Paper: https://arxiv.org/abs/1905.10615
| Website: https://adversarialpolicies.github.io

WaveRL: Training RL agents to perform active damping
----------------------------------------------------
Reinforcement learning is used to train agents to control pistons attached to a bridge to cancel out vibrations.  The bridge is modeled as a one dimensional oscillating system and dynamics are simulated using a finite difference solver.  Agents were trained using Proximal Policy Optimization.  See presentation for environment detalis.

| Authors: Jack Berkowitz
| Email: jackberkowitz88@gmail.com
| GitHub: https://github.com/jaberkow/WaveRL
| Presentation: http://bit.ly/WaveRLslides
