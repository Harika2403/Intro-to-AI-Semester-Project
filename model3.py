import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os
import numpy as np
from torch.distributions import Categorical

class PPONetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(PPONetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        
        # Actor head
        self.actor = nn.Linear(hidden_size, output_size)
        
        # Critic head
        self.critic = nn.Linear(hidden_size, 1)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        
        # Actor: probability distribution over actions
        action_probs = F.softmax(self.actor(x), dim=-1)
        
        # Critic: estimated state value
        state_value = self.critic(x)
        
        return action_probs, state_value
    
    def save(self, file_name='ppo_model.pth'):
        model_folder_path = './model'
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)

        file_name = os.path.join(model_folder_path, file_name)
        torch.save(self.state_dict(), file_name)

class PPOTrainer:
    def __init__(self, model, lr=3e-4, gamma=0.99, clip_param=0.2, ppo_epochs=4, batch_size=64):
        self.lr = lr
        self.gamma = gamma
        self.clip_param = clip_param
        self.ppo_epochs = ppo_epochs
        self.batch_size = batch_size
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr)
    
    def update(self, samples):
        states, actions, old_log_probs, rewards, next_states, dones = samples
        
        # Convert to tensors
        states = torch.FloatTensor(np.array(states))
        actions = torch.LongTensor(np.array(actions))
        old_log_probs = torch.FloatTensor(np.array(old_log_probs))
        rewards = torch.FloatTensor(np.array(rewards)).unsqueeze(-1)
        next_states = torch.FloatTensor(np.array(next_states))
        dones = torch.FloatTensor(np.array(dones)).unsqueeze(-1)
        
        # Calculate returns and advantages
        with torch.no_grad():
            _, values = self.model(states)
            _, next_values = self.model(next_states)
            
            deltas = rewards + self.gamma * next_values * (1 - dones) - values
            advantages = torch.zeros_like(deltas)
            advantage = 0
            for t in reversed(range(len(deltas))):
                advantage = deltas[t] + self.gamma * advantage * (1 - dones[t])
                advantages[t] = advantage
            
            returns = advantages + values
        
        # Optimize policy for K epochs
        for _ in range(self.ppo_epochs):
            # Get current policy and value estimates
            action_probs, values = self.model(states)
            dist = Categorical(action_probs)
            log_probs = dist.log_prob(actions)
            
            # Calculate ratio (pi_theta / pi_theta_old)
            ratios = torch.exp(log_probs - old_log_probs)
            
            # Calculate surrogate losses
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.clip_param, 1 + self.clip_param) * advantages
            actor_loss = -torch.min(surr1, surr2).mean()
            
            # Critic loss
            critic_loss = F.mse_loss(values, returns)
            
            # Entropy bonus
            entropy = dist.entropy().mean()
            
            # Total loss
            loss = actor_loss + 0.5 * critic_loss - 0.01 * entropy
            
            # Update
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        
        return loss.item()