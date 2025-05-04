# Model.py
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os
import numpy as np
from torch.distributions import Categorical

# Model.py
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os
import numpy as np
from torch.distributions import Categorical

class A3CNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(A3CNetwork, self).__init__()
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
    
    def save(self, file_name='a3c_model.pth'):
        model_folder_path = './model'
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)

        file_name = os.path.join(model_folder_path, file_name)
        torch.save(self.state_dict(), file_name)

class A3CTrainer:
    def __init__(self, model, lr, gamma, beta=0.01):
        self.lr = lr
        self.gamma = gamma
        self.beta = beta  # entropy coefficient
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr)
    
    def train_step(self, states, actions, rewards, next_states, dones):
        # Convert to tensors
        states = torch.FloatTensor(np.array(states))
        actions = torch.LongTensor(np.array(actions))
        rewards = torch.FloatTensor(np.array(rewards)).unsqueeze(-1)  # Add dimension
        next_states = torch.FloatTensor(np.array(next_states))
        dones = torch.FloatTensor(np.array(dones)).unsqueeze(-1)  # Add dimension
        
        # Get current policy and value estimates
        action_probs, state_values = self.model(states)
        
        # Get next state values
        with torch.no_grad():
            _, next_state_values = self.model(next_states)
        
        # Calculate returns and advantages
        returns = rewards + self.gamma * next_state_values * (1 - dones)
        advantages = returns - state_values
        
        # Calculate actor (policy) loss
        dist = Categorical(action_probs)
        log_probs = dist.log_prob(actions)
        actor_loss = -(log_probs * advantages.detach()).mean()
        
        # Calculate critic (value) loss - ensure same shape
        critic_loss = F.mse_loss(state_values, returns)
        
        # Calculate entropy (for exploration)
        entropy = dist.entropy().mean()
        
        # Total loss
        total_loss = actor_loss + critic_loss - self.beta * entropy
        
        # Backpropagation
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()
        
        return total_loss.item()