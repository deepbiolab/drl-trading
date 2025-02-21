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
    # test model
    q_net = QNetwork(state_size=8)  # action_size defaults to 3 (buy, sell, hold)
    # fake input, batch size 4
    states = torch.rand((4, 8))
    # fake output
    output = q_net(states)
    expected_shape = torch.Size([4, 3])
    try:
        assert output.shape == expected_shape
    except AssertionError:
        print(f"Assertion failed: Expected output shape {expected_shape}, but got {output.shape}")
