# INTELLIGENT SNAKE AGENT USING DEEP REINFORCEMENT LEARNING
## A Comparative Study of Model-Free RL Algorithms

### Project Overview
This project aims to develop an intelligent agent that can effectively play the classic Snake game using state-of-the-art Deep Reinforcement Learning (DRL) techniques. The primary focus is on implementing and comparing model-free RL algorithms such as Deep Q-Network (DQN), Dueling DQN, Asynchronous Advantage Actor-Critic (A3C), Double DQN, Policy Gradient (PG), Proximal Policy Optimization (PPO), and Soft Actor-Critic (SAC).

### Objective
- Develop an AI agent capable of playing the Snake game using Deep Reinforcement Learning algorithms.
- Compare the performance metrics, including score, efficiency, and stability, across various reinforcement learning techniques.

### Related Literature
Reinforcement learning (RL) has been widely applied in various game environments, including the classic Snake game, with notable contributions such as Deep Q-Network (DQN) agents excelling in navigation and obstacle avoidance [3]. The use of model-free algorithms like DQN has been extended in multiple studies, where enhancements like Double DQN and Dueling DQN reduce overestimation bias and improve performance stability [1] [2]. Moreover, approaches like Asynchronous Advantage Actor-Critic (A3C) and Proximal Policy Optimization (PPO) have been evaluated for their training speed and stability in dynamic environments, offering insights into the trade-offs between training time and convergence [2]. This work builds upon these foundations by comparing various RL algorithms, including DQN, A3C, and PPO, in the context of the Snake game, aiming to assess performance in terms of efficiency, stability, and training time.

### Game Setup
- **Environment**: The Snake game is played on a grid where the snake moves and grows by eating food.
- **State Representation**: Includes the snake’s position, direction, and proximity to obstacles such as walls or its own body.
- **Actions**: The agent can move in four directions: up, down, left, or right.

### Algorithms Used
- **Deep Q-Network (DQN)**: A neural network-based approach to approximate Q-values, using experience replay and target networks for stable training.
- **Asynchronous Advantage Actor-Critic (A3C)**: An on-policy RL algorithm using parallel agents for faster, more stable training. Combines policy gradient (actor) and value function (critic).
- **Policy Gradient (PG)**: Directly optimizes the policy using gradient ascent to maximize the expected return.
- **Proximal Policy Optimization (PPO)**: Uses a clipped objective to avoid large policy updates and ensure stable training.
- **Soft Actor-Critic (SAC)**: Combines Q-learning and policy gradient methods to maximize entropy and balance exploration-exploitation.
- **Dueling DQN**: An extension of DQN that separates the value and advantage streams to improve Q-value estimation.
- **Double DQN**: Modifies DQN by reducing overestimation of Q-values through a second Q-network.

### Key Components
- **Agent**: Implements the RL algorithm and handles training workflows.
- **Game Environment**: Contains the logic for game behavior, snake movement, and collision detection.
- **Model**: Defines the neural network architecture used for action-value approximation.

### Implementation Framework
#### Libraries Used:
- **PyTorch**: For building and training neural networks.
- **Pygame**: For rendering the game environment and simulating gameplay.
- **NumPy**: For numerical operations and handling data structures.

#### Files Overview:
- **game.py**: Game environment setup and logic for the Snake game.
- **agent.py**: Contains the RL algorithms and training logic.
  - A3C
  - DQN
  - Policy Gradient (PG)
  - Proximal Policy Optimization (PPO)
  - Soft Actor-Critic (SAC)
  - Dueling DQN
  - Double DQN
- **model.py**: Defines the neural network model used for approximating Q-values.
  - A3C
  - DQN
  - Policy Gradient (PG)
  - Proximal Policy Optimization (PPO)
  - Soft Actor-Critic (SAC)
  - Dueling DQN
  - Double DQN
- **snake_human_game.py**: Manual control version of the game for testing and comparison.

### Training & Evaluation Metrics
- **Average Score**: Measures consistency across episodes.
- **Maximum Score**: Indicates peak performance.
- **Training Time**: Measures efficiency of each algorithm.
- **Convergence Speed**: How quickly the model stabilizes.
- **Stability**: Variance in scores over time.

### Performance Comparisons

| Algorithm        | Max Score | Average Score | Training Time | Convergence Speed | Stability       |
|------------------|-----------|---------------|---------------|-------------------|-----------------|
| DQN              | 78        | 29.2          | Moderate      | Fast              | High variance   |
| A3C              | 59        | 10.06         | Fast          | Fast              | Low variance    |
| Dueling DQN      | 59        | 12.4          | Slow          | Moderate          | High stability  |
| Double DQN       | 56        | 8.3           | Slow          | Moderate          | Low stability   |
| SAC              | 0.4       | 0.3           | Slow          | Very slow         | Very low        |
| PPO              | 0.6       | 0.4           | Slow          | Very slow         | Low             |

### Conclusion
- **Best Performing Algorithm**: DQN exhibited the highest max score but had high variance in results.
- **Most Balanced**: A3C showed a balanced performance, with stable scores and faster convergence.
- **Underperformers**: SAC, PPO, and Policy Gradient failed to converge within 500 episodes, showing poor performance.

### References
1. D. Ray, A. Ghosh, M. Ojha, and K. P. Singh, "Deep Q-Snake: An Intelligent Agent Mastering the Snake Game with Deep Reinforcement Learning," 2024 IEEE Region 10 Conference (TENCON), 2024, pp. 1465–1472, doi: 10.1109/TENCON61640.2024.10903025.
2. A. del Rio, D. Jimenez, and J. Serrano, "Comparative Analysis of A3C and PPO Algorithms in Reinforcement Learning: A Survey on General Environments," IEEE Access, vol. 12, pp. 146795–146812, Oct. 2024, doi: 10.1109/ACCESS.2024.3472473.
3. J. Wang, D. Xue, J. Zhao, W. Zhou, and H. Li, "Mastering the Game of 3v3 Snakes with Rule-Enhanced Multi-Agent Reinforcement Learning," 2022 IEEE Conference on Games (CoG), Beijing, China, pp. 229–236, Aug. 2022, doi: 10.1109/CoG51982.2022.9893608.
4. https://github.com/patrickloeber/snake-ai-pytorch

### Future Work
- **Algorithm Extensions**: Further work will involve improving training stability, tuning hyperparameters, and exploring advanced algorithms such as Dueling DQN and SAC.
- **Real-time Testing**: Conduct real-time testing of agent performance with varying complexity in Snake environments.
