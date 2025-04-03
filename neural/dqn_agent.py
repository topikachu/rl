import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from collections import deque
import os


class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, 32)
        self.fc2 = nn.Linear(32, 32)
        self.fc3 = nn.Linear(32, action_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

class DQNAgent:
    def __init__(self, state_size, action_size, env, model_path="dqn_model.pth", save_interval=50):
        self.env = env
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=1000)
        self.gamma = 0.9  # Discount rate
        self.epsilon = 1.0  # Exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = DQN(state_size, action_size).to(self.device)
        self.target_model = DQN(state_size, action_size).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.criterion = nn.MSELoss()

        self.model_path = model_path  # File path for saving/loading
        self.save_interval = save_interval  # Save every N episodes
        self.episodes = 0  # Track number of episodes

        # Load the model if it exists
        self.load()

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, game_state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)

        state = self.env.process_observation(game_state)
        with torch.no_grad():
            act_values = self.model(state)
        return torch.argmax(act_values).item()

    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return
        minibatch = random.sample(self.memory, batch_size)
        
        states, actions, rewards, next_states, dones = zip(*minibatch)
        
        states = self.env.process_observation_batch(states).to(self.device)
        next_states = self.env.process_observation_batch(next_states).to(self.device)
        actions = torch.tensor(actions, dtype=torch.long).to(self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        dones = torch.tensor(dones, dtype=torch.float32).to(self.device)
    
        # Compute Q values for current states
        q_values = self.model(states)
        current_q_values = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
    
        # Compute Q values for next states
        with torch.no_grad():
            next_q_values = self.target_model(next_states).max(1)[0]
    
        # Compute target Q values
        target_q_values = rewards + (1 - dones) * self.gamma * next_q_values
    
        # Compute loss
        loss = self.criterion(current_q_values, target_q_values)
    
        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
    
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        self.episodes += 1
        if self.episodes % self.save_interval == 0:
            self.save()

    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def load(self):
        """Load the model if the file exists."""
        if os.path.exists(self.model_path):
            self.model.load_state_dict(torch.load(self.model_path))
            self.update_target_model()
            print(f"Model loaded from {self.model_path}")

    def save(self):
        """Save the model to a file."""
        torch.save(self.model.state_dict(), self.model_path)
        print(f"Model saved to {self.model_path}")