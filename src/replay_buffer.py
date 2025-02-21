"""
Experience Replay Buffer Implementation for Deep Q-Learning

This module implements a replay buffer for storing and sampling experiences
in reinforcement learning, particularly for Deep Q-Learning algorithms.
The buffer stores transitions (state, action, reward, next_state, done)
and enables random sampling for breaking correlations in sequential data.

Author: Tim Lin
Organization: DeepBioLab
License: MIT License
"""

import torch
import random
import numpy as np
from collections import deque, namedtuple


class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, buffer_size, batch_size, seed, device=None):
        """Initialize a ReplayBuffer object.

        Params
        ======
            batch_size (int): size of each training batch
            buffer_size (int): maximum size of buffer
            seed (int): random seed
        """
        self.batch_size = batch_size
        self.memory = deque(maxlen=buffer_size)
        self.experience = namedtuple(
            "Experience",
            field_names=["state", "action", "reward", "next_state", "done"],
        )
        self.seed = random.seed(seed)
        self.device = device

    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)

    def _to_tensor(self, data, dtype=torch.float):
        """Convert numpy array to tensor with specified dtype and device in one operation"""
        return torch.from_numpy(np.vstack(data)).to(device=self.device, dtype=dtype)

    def sample(self):
        """Randomly sample a batch of experiences from memory."""

        experiences = random.sample(self.memory, k=self.batch_size)
        experiences = [e for e in experiences if e is not None]
        states, actions, rewards, next_states, dones = zip(*experiences)

        states = self._to_tensor(states)
        actions = self._to_tensor(actions, dtype=torch.long)
        rewards = self._to_tensor(rewards)
        next_states = self._to_tensor(next_states)
        dones = self._to_tensor(dones, dtype=torch.uint8)

        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)
