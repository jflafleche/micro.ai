# micro.ai
**Creating the world's first autonomous micro-robot!**

# Introduction
The goal of this project is to investigate different Reinforcement Learning (RL) architectures and techniques against scenarios that have relevance to the field of micro robotics.

This project is specifically concerned with the control of magnetic micro and nano-robots. These robots consist of magnetic material that are controlled by surrounding the workspace
with energized coils of wire to generate a magnetic field that acts on the
robot. This project makes the simplification of assuming that the robot only moves in 2 dimensions.

# Scenarios
## Sorting Task
A magnetic micro-robot must sort particles to the left and right sides of the workspace depending on each particle's class.

- Solved with DDPG and Experience Replay.

![Sorting Example](/media/images/sorting_example1.png?raw=true "Optional Title")

## Assembly Task
In progress.

# Brains
## Dueling DQN
Temporarily replaced by DDPG to research a scenario with a continuous action space.

## DDPG
Deep Deterministic Policy Gradient (DDPG) [[1]](https://arxiv.org/abs/1509.02971) uses a model-free, actor-critic algorithm that can successfully learn control policies operating over a continuous action space.

# Memories
## Experience Replay
Experience Replay stores past agent experiences and randomly samples from them to perform network updates. First introduced in [[2]](http://www.dtic.mil/docs/citations/ADA261434).

## Prioritized Experience Replay
In progress. (improving efficiency)

## A3C
In progress.

# References:

[[1]](https://arxiv.org/abs/1509.02971) Lillicrap, Timothy P., et al. "Continuous control with deep reinforcement learning." arXiv preprint arXiv:1509.02971 (2015).

[[2]](http://www.dtic.mil/docs/citations/ADA261434) Lin, Long-Ji. Reinforcement learning for robots using neural networks. No. CMU-CS-93-103. Carnegie-Mellon Univ Pittsburgh PA School of Computer Science, 1993.

[[3]](http://www.nowpublishers.com/article/Details/ROB-023) Diller, Eric, and Metin Sitti. "Micro-scale mobile robotics." Foundations and Trends® in Robotics 2.3 (2013): 143-259.

# Reinforcement Learning Resources:
https://medium.com/@awjuliani/simple-reinforcement-learning-with-tensorflow-part-4-deep-q-networks-and-beyond-8438a3e2b8df#.yit72xseu

http://cs.stanford.edu/people/karpathy/reinforcejs/waterworld.html

http://pemami4911.github.io/blog/2016/08/21/ddpg-rl.html

https://jaromiru.com/2016/10/03/lets-make-a-dqn-implementation/