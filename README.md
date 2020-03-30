# udacity-deeprl-continuous-control
Continuous Control problem using Reinforcement Learning

# Project Details

![](/images/unity-wide.png)
## Unity ML-Agents
**Unity Machine Learning Agents (ML-Agents)** is an open-source Unity plugin that enables games and simulations to serve as environments for training intelligent agents.

For game developers, these trained agents can be used for multiple purposes, including controlling [NPC](https://en.wikipedia.org/wiki/Non-player_character) behavior (in a variety of settings such as multi-agent and adversarial), automated testing of game builds and evaluating different game design decisions pre-release.

In this course, you will use Unity's rich environments to design, train, and evaluate your own deep reinforcement learning algorithms. You can read more about ML-Agents by perusing the [GitHub repository](https://github.com/Unity-Technologies/ml-agents).

## The Environment

For this project, you will work with the [Reacher](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#reacher) environment.

<p align="center">
  <img src="/images/reacher.gif" />
</p>

In this environment, a double-jointed arm can move to target locations. A reward of +0.1 is provided for each step that the agent's hand is in the goal location. Thus, the goal of your agent is to maintain its position at the target location for as many time steps as possible.

The observation space consists of 33 variables corresponding to position, rotation, velocity, and angular velocities of the arm. Each action is a vector with four numbers, corresponding to torque applicable to two joints. Every entry in the action vector should be a number between -1 and 1.

### Distributed Training
* For this project, we will provide you with 20 identical agents, each with its own copy of the environment

This version is useful for algorithms like [PPO](https://arxiv.org/abs/1707.06347), [A3C](https://arxiv.org/abs/1602.01783), and [D4PG](https://arxiv.org/abs/1804.08617) that use multiple (non-interacting, parallel) copies of the same agent to distribute the task of gathering experience.

### Solving the Environment

The barrier for solving the second version of the environment is slightly different, to take into account the presence of many agents. In particular, your agents must get an average score of +30 (over 100 consecutive episodes, and over all agents). Specifically,

* After each episode, we add up the rewards that each agent received (without discounting), to get a score for each agent. This yields 20 (potentially different) scores. We then take the average of these 20 scores.
* This yields an **average score** for each episode (where the average is over all 20 agents).

# Getting Started

## Dependencies

To set up your python environment to run the code in this repository, follow the instructions below.

1. Create (and activate) a new environment with Python 3.6.

	- __Windows__: 
	```bash
	conda create --name drlnd python=3.6 
	activate drlnd
	```

2. Install pytorch >= 1.4.0

3. Follow the instructions in [this repository](https://github.com/openai/gym) to perform a minimal install of OpenAI gym.  
	- Next, install the **classic control** environment group by following the instructions [here](https://github.com/openai/gym#classic-control).
	- Then, install the **box2d** environment group by following the instructions [here](https://github.com/openai/gym#box2d).
	
4. Clone the repository (if you haven't already!), and navigate to the `python/` folder.  Then, install several dependencies.
```bash
git clone https://github.com/erichans/udacity-deeprl-continuous-control.git
cd udacity-deeprl-continuous-control/python
pip install .
```
# Instructions

## Train the Agent
```bash
python train.py
```
You can tune the model by changing the following hyperparameters in following files (default values below):

### train.py
* TOTAL_EPISODES = 200

### agent.py
* BUFFER_SIZE = 1.000.000
* BATCH_SIZE = 64 (times the number of agents. i.e 64 x 20 agents -> Batch size: 1280)
* GAMMA = .99 (discount factor)
* TAU = 1e-3 (soft update from local actor and critic network parameters to their respective target network parameters)
* LR_ACTOR = 1e-4 (Actor local learning rate)
* LR_CRITIC = 1e-3 (Critic local learning rate)
* WEIGHT_DECAY = 0 (L2 weight decay for the Actor local)
