# TrafficSimulator

<p align="center">
  <strong>A simplified traffic simulation environment for studying emergent driving behaviors with Reinforcement Learning</strong>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.x-blue?logo=python" alt="Python">
  <img src="https://img.shields.io/badge/Reinforcement%20Learning-DQN-orange" alt="Reinforcement Learning">
  <img src="https://img.shields.io/badge/Environment-Gymnasium-green" alt="Gymnasium">
  <img src="https://img.shields.io/badge/Simulation-2D%20Grid--World-purple" alt="2D Grid World">
</p>

---

## Overview

**TrafficSimulator** is a custom 2D traffic simulation environment developed to investigate how driving behaviors can emerge in **Reinforcement Learning (RL) agents** under different traffic conditions.

The project was developed as part of a Master's thesis in Engineering in Computer Science and focuses on a specific research question:

> **Can a reinforcement learning agent, initially trained to follow basic traffic rules, spontaneously develop non-compliant behaviors as an adaptive response to congestion and the behavior of surrounding vehicles?**

Unlike traditional autonomous-driving simulators, which generally prioritize safe and rule-compliant behavior, this environment is intentionally simplified and allows traffic-rule violations to occur while still penalizing them.

The goal is **not to train an agent to break traffic rules**, but rather to investigate whether non-compliant behaviors can emerge naturally from the interaction between the learned policy, environmental constraints, and traffic conditions.

---

## Environment

The environment is a **2D grid world** representing a simplified urban road network.

It contains:

* one-way and two-way roads;
* intersections controlled by traffic lights;
* static obstacles;
* a predefined trajectory for the agent vehicle;
* dynamically moving NPC vehicles;
* traffic-rule constraints.

The agent is a car that must follow a given path while respecting a set of constraints as much as possible.

### Layered Map

The environment uses a layered grid representation in which different layers describe different aspects of the road network and its current state.

The mentioned layers describe:
* allowed driving directions;
* traffic lights;
* static/dynamic vehicles;
* the agent's trajectory.

---

## Partial Observability

The agent does not have access to the complete environment.

Instead, it observes a local **Field of View (FOV)** centered around its current position.

The observation contains four main layers:

* **Trajectory** — visible portion of the target path;
* **Obstacles** — static and dynamic obstacles;
* **Traffic lights** — visible traffic lights and their current states;
* **Allowed directions** — legal movement directions for the observed road cells.

This creates a partially observable driving problem.

### Alert Area

In addition to the local FOV, the environment provides an **alert area** for detecting distant NPC vehicles.

Vehicles outside the normal FOV but inside the alert area are projected onto the outer boundary of the FOV.

This provides the agent with information about distant traffic while preserving a limited local perception.

The experiments use, among other configurations:

* **5×5 FOV**
* **9×9 alert area**

### Action space

The agent has five possible actions:

| Action  | Description       |
| ------- | ----------------- |
| `RIGHT` | Move right        |
| `UP`    | Move up           |
| `LEFT`  | Move left         |
| `DOWN`  | Move down         |
| `STAY`  | Remain stationary |

---

## Reinforcement Learning

The driving agent is trained using **Deep Q-Network (DQN)**.

At each time step, the agent:

1. observes its local environment (the portion of map corresponding to its field of view);
2. selects an action;
3. receives a reward or penalty;
4. observes the resulting state;
5. updates its policy through experience replay.

The DQN approximates the action-value function and allows the agent to learn which actions are most beneficial in different traffic situations.


---

## Rewards and Constraints

The reward function encourages the agent to make progress along the assigned trajectory while discouraging inefficient and unsafe behavior.

The agent is rewarded for:

* advancing along the trajectory;
* reaching the destination.

Penalties are applied for:

* unnecessary inactivity;
* deviating from the trajectory;
* collisions;
* traffic-light violations;
* driving in a prohibited direction.

A key design choice is that traffic violations are **penalized rather than universally forbidden**.

This allows the agent to choose a non-compliant action when the learned policy estimates that doing so provides a better long-term outcome.

### Constraints

The environment currently includes constraints related to:

* **Collision**
* **Traffic lights**
* **Allowed driving directions**

Collisions and traffic-light violations can terminate an episode, while lane-direction violations are penalized according to the environment configuration.

---

## NPC Vehicles

NPC vehicles are used to create increasingly challenging traffic conditions.

Different NPC configurations can be used to investigate how surrounding traffic affects the learning and behavior of the agent.

This makes it possible to study how congestion and the behavior of other drivers influence the emergence of adaptive strategies.

---

## Experiments

## Experiments

The repository contains the code used to train, evaluate, and analyze reinforcement learning agents in different traffic scenarios.

The available implementations include:

* single-agent training;
* fine-tuning with NPC vehicles;
* different NPC configurations;
* movement-aware observations;
* training and evaluation utilities;
* plotting and analysis tools.

Pre-trained policies and experiment configurations are provided in the `weights/` and `config/` directories where applicable.

---

## Project Structure

```text
TrafficSimulator/
│
├── agent/          # DQN agent, neural network and replay buffer
├── assets/         # Images and assets used by the simulator
├── config/         # Environment, agent and training configurations
├── constraints/    # Traffic-rule and safety constraints
├── env/            # Custom Gymnasium traffic environment
├── plot/           # Functions for plotting training and evaluation metrics
├── tests/          # Tests and experimental/debugging scripts
├── train/          # Training and experiment scripts
├── utils/          # Auxiliary functions
├── weights/        # Saved trained policies
│
└── README.md
```

---

## Installation

Clone the repository:

```bash
git clone https://github.com/vinbeatrice/TrafficSimulator.git
cd TrafficSimulator
```

Install the required Python packages:

```bash
pip install numpy pygame gymnasium torch matplotlib
```

A virtual environment is recommended:

```bash
python -m venv .venv
```

Activate it on Windows:

```bash
.venv\Scripts\activate
```

or on Linux/macOS:

```bash
source .venv/bin/activate
```

Then install the dependencies:

```bash
pip install numpy pygame gymnasium torch matplotlib
```

---

## Running the Project

The main training and experiment scripts are located in the `train/` directory.

Before starting an experiment, check the configuration files in:

```text
config/
```

The configuration controls parameters such as:

* number of training episodes;
* maximum episode length;
* learning rate;
* discount factor;
* replay-buffer size;
* epsilon exploration parameters;
* FOV and alert-area dimensions;
* number of NPCs;
* reward and penalty values.

Trained policies are stored in:

```text
weights/
```

Plots and evaluation metrics can be generated using the utilities in:

```text
plot/
```

> **Note:** the exact training script and configuration to use depend on the experiment being reproduced. Refer to the corresponding files in `train/` and `config/`.

---

## Research Context

This repository accompanies a Master's thesis on **Reinforcement Learning and emergent driving behaviors**.

The work investigates the relationship between:

```text
Traffic constraints
       +
Surrounding vehicles
       +
Learned policy
       ↓
Emergent driving behavior
```

The central idea is to treat traffic violations not exclusively as learning failures, but also as potential **adaptive responses to environmental pressure**.

The environment is therefore intentionally simpler than high-fidelity autonomous-driving simulators. This makes it possible to focus on the agent's decision-making behavior rather than on realistic physics or visual simulation.

---

## Future Work

Possible extensions of the project include:

* multi-agent reinforcement learning;
* more complex road networks;
* richer NPC behaviors;
* additional traffic rules and constraints;
* more advanced RL algorithms;
* improved modeling of vehicle dynamics;
* more realistic traffic scenarios;
* further analysis of emergent non-compliant behaviors.

