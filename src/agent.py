import random
import numpy as np

import torch
import torch.nn.functional as F
import torch.optim as optim

from .model import QNetwork
from .replay_buffer import ReplayBuffer


class Agent:
    """Interacts with and learns from the environment."""

    def __init__(
        self,
        num_features,
        window_size,
        test_mode=False,
        buffer_size=int(1e5),
        batch_size=64,
        gamma=0.99,
        alpha=1e-3,
        lr=5e-4,
        update_step=4,
        seed=42,
        device=None,
    ):
        """Initialize an Agent object.

        Key Features:
            - State space: Concatenated window of features (window_size * num_features)
            - Action space: 3 discrete actions (0=hold, 1=buy, 2=sell)
            - Experience replay: Stores and samples past experiences for stable learning
            - Target network: Updated softly for stable training

        Params
        ======
            buffer_size (int): Maximum size of experience replay buffer (default: 1e5)
            batch_size (int): Size of each training batch (default: 64)
            gamma (float): Discount factor for future rewards (default: 0.99)
            alpha (float): Soft update interpolation parameter (default: 1e-3)
            lr (float): Learning rate for optimizer (default: 5e-4)
            update_step (int): Frequency of network updates (default: 4)
            test_mode (bool): Flag for switching between training and testing behavior
        """
        self.seed = random.seed(seed)
        self.test_mode = test_mode
        self.device = device

        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.lr = lr
        self.gamma = gamma
        self.alpha = alpha
        self.update_step = update_step

        self.state_size = window_size * num_features
        self.action_size = 3

        # Q-Network
        self.Q = QNetwork(self.state_size, self.action_size, seed).to(device)
        self.Q_target = QNetwork(self.state_size, self.action_size, seed).to(device)
        self.optimizer = optim.Adam(self.Q.parameters(), lr=self.lr)

        # Replay memory
        self.memory = ReplayBuffer(self.buffer_size, self.batch_size, seed)
        # Initialize time step (for updating every update step)
        self.t_step = 0

    def __repr__(self):
        return (
            f"Q Network Arch: {self.Q}\n"
            f"State space size: {self.state_size}\n"
            f"Action space size: {self.action_size}\n"
            f"Current Memory size: {len(self.memory)}"
        )

    def step(self, state, action, reward, next_state, done):
        # Save experience in replay memory
        self.memory.add(state, action, reward, next_state, done)

        # Learn every update_step time steps.
        self.t_step = (self.t_step + 1) % self.update_step
        if self.t_step == 0:
            # If enough samples are available in memory, get random subset and learn
            if len(self.memory) > self.batch_size:
                experiences = self.memory.sample()
                loss = self.learn(experiences)
                return loss

    def select_action(self, state, epsilon=0.0):
        """Returns actions for given state as per current policy.

        Params
        ======
            state (array_like): current state
            eps (float): epsilon, for epsilon-greedy action selection
        """
        state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        self.Q.eval()
        with torch.no_grad():
            actions = self.Q(state)

        if self.test_mode:
            return np.argmax(actions.cpu().data.numpy())

        self.Q.train()
        # Epsilon-greedy action selection
        if random.random() <= epsilon:
            return random.choice(np.arange(self.action_size))
        else:
            return np.argmax(actions.cpu().data.numpy())

    def learn(self, experiences):
        """Update value parameters using given batch of experience tuples.

        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples
            gamma (float): discount factor
        """
        self.optimizer.zero_grad()

        # get experiences
        states, actions, rewards, next_states, dones = experiences

        # compute td targets using target network
        with torch.no_grad():
            Q_targets_next = torch.max(
                self.Q_target(next_states), dim=-1, keepdim=True
            )[0]
            Q_targets = rewards + (1 - dones) * self.gamma * Q_targets_next

        # compute curr values using local network
        Q_expected = torch.gather(self.Q(states), dim=-1, index=actions)

        # compute mean squared loss using td error
        loss = F.mse_loss(Q_expected, Q_targets)
        loss.backward()

        # update local network parameters
        self.optimizer.step()

        # update target network parameters
        self.soft_update()

        return loss

    def soft_update(self):
        """Soft update model parameters.
        θ_target = alpha*θ + (1 - alpha)*θ_target
        =>
        θ_target = θ_target + alpha*(θ - θ_target)

        Params
        ======
            Q (PyTorch model): weights will be copied from
            Q_target (PyTorch model): weights will be copied to
            alpha (float): interpolation parameter
        """
        for target_param, local_param in zip(
            self.Q_target.parameters(), self.Q.parameters()
        ):
            target_param.data.copy_(
                target_param.data + self.alpha * (local_param.data - target_param.data)
            )

    def hard_update(self):
        """Hard update: θ_target = θ"""
        for target_param, local_param in zip(
            self.Q_target.parameters(), self.Q.parameters()
        ):
            target_param.data.copy_(local_param.data)

    def save(self, filename):
        """Save model parameters."""
        torch.save(self.Q.state_dict(), filename)

    def load(self, filename):
        """Load model parameters."""
        checkpoint = torch.load(filename)
        self.Q.load_state_dict(checkpoint)