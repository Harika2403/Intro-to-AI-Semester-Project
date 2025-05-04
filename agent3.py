import torch
import random
import numpy as np
from collections import deque
from torch.distributions import Categorical
from game import SnakeGameAI, Direction, Point
from model3 import PPONetwork, PPOTrainer
from helper import plot
import time
from metrics import MetricsTracker

MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001
GAMMA = 0.9
CLIP_PARAM = 0.2
PPO_EPOCHS = 4

class PPOAgent:
    def __init__(self):
        self.n_games = 0
        self.model = PPONetwork(11, 256, 3)
        self.trainer = PPOTrainer(self.model, lr=LR, gamma=GAMMA, clip_param=CLIP_PARAM, ppo_epochs=PPO_EPOCHS)
        self.memory = deque(maxlen=MAX_MEMORY)
        
    def get_state(self, game):
        # Same state representation as before
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
    
    def remember(self, state, action, old_log_prob, reward, next_state, done):
        self.memory.append((state, action, old_log_prob, reward, next_state, done))
    
    def get_action(self, state):
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            action_probs, _ = self.model(state_tensor)
        
        dist = Categorical(action_probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action.item(), log_prob.item()
    
    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE)
        else:
            mini_sample = self.memory

        states, actions, old_log_probs, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.update((states, actions, old_log_probs, rewards, next_states, dones))
    
    def train_short_memory(self, state, action, old_log_prob, reward, next_state, done):
        self.remember(state, action, old_log_prob, reward, next_state, done)

def train(num_episodes=None, metrics_callback=None):
    metrics = MetricsTracker()
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record = 0
    start_time = time.time()
    
    agent = PPOAgent()
    game = SnakeGameAI()
    
    while True:
        # get old state
        state_old = agent.get_state(game)

        # get move
        action, log_prob = agent.get_action(state_old)
        final_move = [0, 0, 0]
        final_move[action] = 1

        # perform move and get new state
        reward, done, score = game.play_step(final_move)
        state_new = agent.get_state(game)

        # train short memory
        agent.train_short_memory(state_old, action, log_prob, reward, state_new, done)

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

            print('Game', agent.n_games, 'Score', score, 'Record:', record)

            plot_scores.append(score)
            total_score += score
            mean_score = total_score / agent.n_games
            plot_mean_scores.append(mean_score)
            
            if agent.n_games % 100 == 0:
                metrics.plot_metrics("PPO")
                print("Summary stats:", metrics.get_summary_stats())

            plot(plot_scores, plot_mean_scores)
            
            if num_episodes and agent.n_games >= num_episodes:
                break

if __name__ == '__main__':
    train()