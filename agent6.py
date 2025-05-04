import torch
import random
import numpy as np
from collections import deque
from game import SnakeGameAI, Direction, Point
from model6 import DoubleDQN, DoubleDQNTrainer
from helper import plot
import time
from metrics import MetricsTracker

MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001
GAMMA = 0.9
UPDATE_TARGET = 100

class DoubleDQNAgent:
    def __init__(self):
        self.n_games = 0
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.model = DoubleDQN(11, 256, 3)
        self.target_model = DoubleDQN(11, 256, 3)
        self.target_model.load_state_dict(self.model.state_dict())
        self.trainer = DoubleDQNTrainer(self.model, self.target_model, lr=LR, gamma=GAMMA)
        self.memory = deque(maxlen=MAX_MEMORY)
        
    def get_state(self, game):
        head = game.snake[0]
        point_l = Point(head.x - 20, head.y)
        point_r = Point(head.x + 20, head.y)
        point_u = Point(head.x, head.y - 20)
        point_d = Point(head.x, head.y + 20)
        
        dir_l = game.direction == Direction.LEFT
        dir_r = game.direction == Direction.RIGHT
        dir_u = game.direction == Direction.UP
        dir_d = game.direction == Direction.DOWN

        state = [
            # Danger straight
            (dir_r and game.is_collision(point_r)) or 
            (dir_l and game.is_collision(point_l)) or 
            (dir_u and game.is_collision(point_u)) or 
            (dir_d and game.is_collision(point_d)),

            # Danger right
            (dir_u and game.is_collision(point_r)) or 
            (dir_d and game.is_collision(point_l)) or 
            (dir_l and game.is_collision(point_u)) or 
            (dir_r and game.is_collision(point_d)),

            # Danger left
            (dir_d and game.is_collision(point_r)) or 
            (dir_u and game.is_collision(point_l)) or 
            (dir_r and game.is_collision(point_u)) or 
            (dir_l and game.is_collision(point_d)),
            
            # Move direction
            dir_l,
            dir_r,
            dir_u,
            dir_d,
            
            # Food location 
            game.food.x < game.head.x,  # food left
            game.food.x > game.head.x,  # food right
            game.food.y < game.head.y,  # food up
            game.food.y > game.head.y  # food down
            ]

        return np.array(state, dtype=int)
    
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
    
    def get_action(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, 2)
        
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            q_values = self.model(state_tensor)
        return torch.argmax(q_values).item()
    
    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE)
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, dones = zip(*mini_sample)
        loss = self.trainer.train_step(states, actions, rewards, next_states, dones)
        
        # Decay epsilon
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        
        # Update target network
        if self.n_games % UPDATE_TARGET == 0:
            self.target_model.load_state_dict(self.model.state_dict())
            
        return loss
    
    def train_short_memory(self, state, action, reward, next_state, done):
        loss = self.trainer.train_step([state], [action], [reward], [next_state], [done])
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        return loss

def train(num_episodes=None, metrics_callback=None):
    metrics = MetricsTracker()
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record = 0
    start_time = time.time()
    
    agent = DoubleDQNAgent()
    game = SnakeGameAI()
    
    while True:
        # get old state
        state_old = agent.get_state(game)

        # get move
        action = agent.get_action(state_old)
        final_move = [0, 0, 0]
        final_move[action] = 1

        # perform move and get new state
        reward, done, score = game.play_step(final_move)
        state_new = agent.get_state(game)

        # train short memory
        agent.train_short_memory(state_old, action, reward, state_new, done)

        # remember
        agent.remember(state_old, action, reward, state_new, done)

        if done:
            episode_length = game.frame_iteration
            metrics.record_episode(score, episode_length)
            
            if metrics_callback:
                metrics_callback(score, episode_length)
            
            game.reset()
            agent.n_games += 1
            agent.train_long_memory()

            if score > record:
                record = score
                agent.model.save()

            print('Game', agent.n_games, 'Score', score, 'Record:', record, 'Epsilon:', agent.epsilon)

            plot_scores.append(score)
            total_score += score
            mean_score = total_score / agent.n_games
            plot_mean_scores.append(mean_score)
            
            window_size = 100
            if len(plot_scores) >= window_size:
                moving_average = np.mean(plot_scores[-window_size:])
                plot_mean_scores.append(moving_average)

            if agent.n_games % 100 == 0:
                metrics.plot_metrics("Double DQN")
                print("Summary stats:", metrics.get_summary_stats())

            plot(plot_scores, plot_mean_scores)
            
            if num_episodes and agent.n_games >= num_episodes:
                break

def train_for_comparison(num_episodes):
    metrics = {
        'scores': [],
        'episode_lengths': [],
        'timestamps': []
    }
    start_time = time.time()
    
    def callback(score, length):
        metrics['scores'].append(score)
        metrics['episode_lengths'].append(length)
        metrics['timestamps'].append(time.time() - start_time)
    
    train(num_episodes=num_episodes, metrics_callback=callback)
    return metrics

if __name__ == '__main__':
    train()