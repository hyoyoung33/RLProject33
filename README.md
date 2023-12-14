# Taxi-v3 Reinforcement Learning Comparison

## Overview

This project uses the Taxi-v3 environment from OpenAI's gymnasium to compare two reinforcement learning algorithms: Q-learning and SARSA. Both algorithms implement an epsilon-greedy policy, with a focus on different strategies for epsilon decay. The project aims to evaluate the performance of these algorithms in terms of learning efficiency and effectiveness. It also includes visualizations of the learning progress for a more intuitive understanding of each algorithm's behavior over time.

## Algorithms

<li><b>On-policy TD control - SARSA(State-Action-Reward-State-Action):</b> The agent behaves using one policy and tries to improve the same policy.
<li><b>Off-policy TD control - Q-learning:</b> The agent behaves using one policy and tries to improve a different policy.

## Epsilon-Greedy Strategy

The project utilizes an epsilon-greedy policy to balance exploration and exploitation.
Two strategies for epsilon decay are compared:

 <li>Discrete Interval Decay : Epsilon is decreased at fixed intervals 

 <li>Exponential Decay : Epsilon is decreased exponentially, providing a more gradual decrease in exploration as learning progresses.

## Installation

  ```
     pip install -r Requirements_lib.txt
  ```

## File Structure

Taxi_QL.py: Implementation of the Q-learning algorithm.</br>
Taxi_SARSA.py: Implementation of the SARSA algorithm.</br>
Taxi_Q_gamma.py: Hyperparameter tuning of gamma</br>
Taxi_Q_alpha.py: Hyperparameter tuning of alpha</br>
Taxi_Q_ID.py: Implementation of the discrete Interval decay in the ε-greedy strategy</br>
Taxi_QL_ED.py: : Implementation of the exponential decay in the ε-greedy strategy</br>

