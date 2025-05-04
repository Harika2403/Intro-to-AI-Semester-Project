import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import torch.optim as optim
import numpy as np

class DuelingDQN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(DuelingDQN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        
        # Value stream
        self.value_fc = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, 1)
        
        # Advantage stream
        self.advantage_fc = nn.Linear(hidden_size, hidden_size)
        self.advantage = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        
        # Value stream
        value = F.relu(self.value_fc(x))
        value = self.value(value)
        
        # Advantage stream
        advantage = F.relu(self.advantage_fc(x))
        advantage = self.advantage(advantage)
        
        # Combine streams
        q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))
        return q_values
    
    def save(self, file_name='dueling_dqn.pth'):
        model_folder_path = './model'
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)

        file_name = os.path.join(model_folder_path, file_name)
        torch.save(self.state_dict(), file_name)

class DuelingDQNTrainer:
    def __init__(self, model, target_model, lr, gamma):
        self.lr = lr
        self.gamma = gamma
        self.model = model
        self.target_model = target_model
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()
        
    def train_step(self, states, actions, rewards, next_states, dones):
        states = torch.FloatTensor(np.array(states))
        actions = torch.LongTensor(np.array(actions)).unsqueeze(-1)
        rewards = torch.FloatTensor(np.array(rewards)).unsqueeze(-1)
        next_states = torch.FloatTensor(np.array(next_states))
        dones = torch.FloatTensor(np.array(dones)).unsqueeze(-1)
        
        # Current Q values
        current_q = self.model(states).gather(1, actions)
        
        # Target Q values
        with torch.no_grad():
            next_q = self.target_model(next_states).max(1)[0].unsqueeze(-1)
            target_q = rewards + self.gamma * next_q * (1 - dones)
        
        # Compute loss
        loss = self.criterion(current_q, target_q)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item()