import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from collections import deque
import os
from logger_config import get_logger

from typing import Deque, Tuple

from robocode_env import  RobocodeGameState
logger = get_logger(__name__)


class DQN(nn.Module):
    def __init__(self, state_size: int, action_size: int):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, 32)
        self.fc2 = nn.Linear(32, 32)
        self.fc3 = nn.Linear(32, action_size)
        logger.debug(f"DQN model initialized with state_size: {state_size}, action_size: {action_size}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

class DQNAgent:
    def __init__(self, state_size: int, action_size: int, env, writer, model_path: str = "dqn_model.pth", save_interval: int = 1000):
        self.env = env
        self.state_size = state_size
        self.action_size = action_size
        self.memory: Deque[Tuple[RobocodeGameState, int, float, RobocodeGameState, bool]] = deque(maxlen=50000)
        self.gamma = 0.9  # Discount rate
        self.epsilon = 1.0  # Exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay =  0.9995
        self.learning_rate = 0.001
        # Updated device selection logic
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif torch.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")

        self.writer = writer
        self.train_step = 0

        self.model = DQN(state_size, action_size).to(self.device)
        self.target_model = DQN(state_size, action_size).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.criterion = nn.MSELoss()

        self.model_path = model_path
        self.save_interval = save_interval
        self.episodes = 0

        self.load()
        logger.info(f"DQNAgent initialized. Device: {self.device}, Model path: {self.model_path}")

    def remember(self, state: RobocodeGameState, action: int, reward: float, next_state: RobocodeGameState, done: bool) -> None:
        self.memory.append((state, action, reward, next_state, done))
        logger.debug(f"Memory updated. Current size: {len(self.memory)}")

    def act(self, game_state: RobocodeGameState) -> int:
        if np.random.rand() <= self.epsilon:
            action = random.randrange(self.action_size)
            logger.debug(f"Random action chosen: {action}. Epsilon: {self.epsilon}")
            return action

        state = self.env.process_observation(game_state).to(self.device)
        with torch.no_grad():
            act_values = self.model(state)
        action = torch.argmax(act_values).item()
        logger.debug(f"Model-based action chosen: {action}. Epsilon: {self.epsilon}")
        return action

    def replay(self, batch_size: int) -> None:
        if len(self.memory) < batch_size:
            logger.debug(f"Skipping replay. Memory size ({len(self.memory)}) < batch size ({batch_size})")
            return

        minibatch = random.sample(self.memory, batch_size)
        states, actions, rewards, next_states, dones = zip(*minibatch)

        states = self.env.process_observation_batch(states).to(self.device)
        next_states = self.env.process_observation_batch(next_states).to(self.device)
        actions = torch.tensor(actions, dtype=torch.long).to(self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        dones = torch.tensor(dones, dtype=torch.float32).to(self.device)

        q_values = self.model(states)
        current_q_values = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            next_q_values = self.target_model(next_states).max(1)[0]

        target_q_values = rewards + (1 - dones) * self.gamma * next_q_values

        loss = self.criterion(current_q_values, target_q_values)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()

        self.writer.add_scalar('Loss/Train', loss.item(), self.train_step)
        self.train_step += 1

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        self.episodes += 1
        logger.debug(f"Replay performed. Loss: {loss.item()}, Epsilon: {self.epsilon}")

        if self.episodes % self.save_interval == 0:
            self.save()
    def update_target_model(self) -> None:
        self.target_model.load_state_dict(self.model.state_dict())
        logger.info("Target model updated")

    def load(self) -> None:
        if os.path.exists(self.model_path):
            self.model.load_state_dict(torch.load(self.model_path))
            self.update_target_model()
            logger.info(f"Model loaded from {self.model_path}")
        else:
            logger.info(f"No existing model found at {self.model_path}. Starting with a new model.")

    def save(self) -> None:
        torch.save(self.model.state_dict(), self.model_path)
        logger.info(f"Model saved to {self.model_path}")
