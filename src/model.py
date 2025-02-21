"""
Deep Q-Network (DQN) Model Implementation

This module implements a neural network architecture for Deep Q-Learning,
specifically designed for trading environments. The network maps state
observations to Q-values for each possible action (hold, buy, sell).


Author: Tim Lin
Organization: DeepBioLab
License: MIT License
"""

import torch
import torch.nn as nn


class QNetwork(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size=3, seed=42):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action (default=3 for buy, sell, hold)
            seed (int): Random seed
        """
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.Q = nn.Sequential(
            nn.Linear(state_size, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 8),
            nn.ReLU(),
            nn.Linear(8, action_size),  # Linear activation by default
        )

    def forward(self, state):
        """Build a network that maps state -> action values."""
        actions = self.Q(state)
        return actions


if __name__ == "__main__":
    print("Testing QNetwork...")

    # Test 1: Initialization
    state_size = 8
    action_size = 3
    q_net = QNetwork(state_size=state_size)

    # Check network structure
    assert hasattr(q_net, "Q"), "Network should have Q Sequential layer"
    assert len(list(q_net.parameters())) > 0, "Network should have trainable parameters"
    print("✓ Initialization tests passed")

    # Test 2: Forward Pass
    # Test single input
    single_state = torch.rand((1, state_size))
    single_output = q_net(single_state)
    expected_single_shape = torch.Size([1, action_size])
    assert (
        single_output.shape == expected_single_shape
    ), f"Single input shape mismatch. Expected {expected_single_shape}, got {single_output.shape}"

    # Test batch input
    batch_size = 4
    states = torch.rand((batch_size, state_size))
    output = q_net(states)
    expected_batch_shape = torch.Size([batch_size, action_size])
    assert (
        output.shape == expected_batch_shape
    ), f"Batch input shape mismatch. Expected {expected_batch_shape}, got {output.shape}"
    print("✓ Forward pass shape tests passed")
