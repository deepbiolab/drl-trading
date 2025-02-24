"""
Trading Environment
===================

This environment simulates a trading scenario using historical price data and 
technical indicators. The agent can perform three actions:
- Hold (0): Maintain current position
- Buy (1): Purchase one unit of the asset
- Sell (2): Sell one unit of the asset from inventory

The environment provides:
- State: Window of price/indicator differences
- Reward: Profit from successful trades (positive only)
- Info: Dictionary with trade details, inventory, and profits

Usage:
------
```
env = Environment(data, window_size=10)
state = env.reset()
action = agent.select_action(state)
next_state, reward, done, info = env.step(action)
```

Author: Tim Lin
Organization: DeepBioLab
License: MIT License
"""

from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd


def sigmoid(x):
    return 1 / (1 + np.exp(-x))

class Environment:
    """
    Trading environment for reinforcement learning agent.

    Attributes:
    -----------
    data : pd.DataFrame
        DataFrame containing trading data and indicators
    features : np.ndarray
        NumPy array of feature values
    feature_columns : List[str]
        List of feature column names
    feature_map : Dict[str, int]
        Mapping of feature names to array indices
    n_features : int
        Number of features in the data
    n_samples : int
        Total number of time steps in the data
    current_step : int
        Current time step in the episode
    inventory : List[float]
        List of purchase prices for held positions
    total_profit : float
        Cumulative profit from all trades
    window_size : int
        Number of time steps in state observation
    state_size : int
        Size of the state vector (window_size * n_features)
    action_size : int
        Number of possible actions (3: hold, buy, sell)
    """

    def __init__(
        self, data: pd.DataFrame, window_size: int = 10, verbose=False
    ) -> None:
        """
        Initialize the trading environment.

        Parameters:
        -----------
        data : pd.DataFrame
            DataFrame containing trading data and indicators
        window_size : int, default=10
            Number of time steps to include in state observation
        """
        self.verbose = verbose

        # Data initialization
        self.data = data
        self.features = self.data.values
        self.feature_columns = self.data.columns.tolist()
        self.feature_map = self._create_feature_mapping(self.feature_columns)

        # Environment dimensions
        self.n_features = len(self.feature_columns)
        self.n_samples = len(self.data)

        # Trading state variables
        self.current_step: Optional[int] = None
        self.inventory: Optional[List[float]] = None
        self.total_profit: Optional[float] = None

        # Environment configuration
        self.window_size = window_size
        self.state_size = self.window_size * self.n_features
        self.action_size = 3  # hold (0), buy (1), sell (2)

        self.total_winners = 0.0
        self.total_losers = 0.0
        self.states_buy = []
        self.states_sell = []

        # Initialize environment state
        self.reset()

    def reset(self) -> np.ndarray:
        """
        Reset the environment to initial state.

        Returns:
        --------
        np.ndarray
            Initial state observation
        """
        self.current_step = 0
        self.inventory = []
        self.total_profit = 0.0
        self.total_winners = 0.0
        self.total_losers = 0.0
        self.states_buy = []
        self.states_sell = []
        return self._get_state(self.current_step)

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        Execute trading action and return new state, reward, and info.

        Parameters:
        -----------
        action : int
            Trading action (0: hold, 1: buy, 2: sell)

        Returns:
        --------
        next_state : np.ndarray
            Next state observation
        reward : float
            Reward for the action (profit from trade if any)
        done : bool
            Whether episode is finished
        info : dict
            Additional information including:
            - action: Executed action type
            - trade_profit: Profit from trade (if sold)
            - total_profit: Cumulative profit
            - inventory: Number of held positions
        """
        reward = 0.0
        info = {}

        current_price = self._get_current_price()

        if action == 1:  # Buy
            self.inventory.append(current_price)
            self.states_buy.append(self.current_step)
            info["action"] = "buy"
            info["buy_price"] = current_price.item()
            if self.verbose:
                print(f"Buy: {self.format_price(current_price.item())}")

        elif action == 2 and len(self.inventory) > 0:  # Sell
            bought_price = self.inventory.pop(0)
            trade_profit = current_price - bought_price
            reward = max(trade_profit, 0)  # Only positive rewards
            self.total_profit += trade_profit.item()

            # Track trade statistics
            if trade_profit >= 0:
                self.total_winners += trade_profit.item()
            else:
                self.total_losers += trade_profit.item()

            self.states_sell.append(self.current_step)

            # Update info dictionary
            info["action"] = "sell"
            info["sell_price"] = current_price.item()
            info["bought_price"] = bought_price.item()
            info["trade_profit"] = trade_profit.item()

            if self.verbose:
                print(
                    f"Sell: {self.format_price(current_price.item())} | "
                    f"Profit: {self.format_price(trade_profit.item())}"
                )
        else:
            info["action"] = "hold"

        # Update state
        self.current_step += 1
        done = 1 if self.current_step >= self.n_samples - 1 else 0

        # Get next state observation
        next_state = self._get_state(self.current_step)

        # Add additional info
        info.update(
            {
                "current_step": self.current_step,
                "current_price": current_price.item(),
                "total_profit": self.total_profit,
                "inventory_size": len(self.inventory),
                "total_winners": self.total_winners,
                "total_losers": self.total_losers,
                "states_buy": self.states_buy.copy(),
                "states_sell": self.states_sell.copy(),
            }
        )

        if self.inventory:
            info["inventory_prices"] = self.inventory.copy()
            info["avg_buy_price"] = np.mean(self.inventory).item()

        return next_state, reward, done, info

    def get_feature_value(self, feature_name: str, step: Optional[int] = None) -> float:
        """
        Get the value of a specific feature at a given step.

        Parameters:
        -----------
        feature_name : str
            Name of the feature to retrieve
        step : int, optional
            Step at which to get the feature value. If None, uses current_step

        Returns:
        --------
        float
            Value of the requested feature

        Raises:
        -------
        KeyError
            If requested feature is not found in the data
        """
        if step is None:
            step = self.current_step

        try:
            feature_idx = self.feature_map[feature_name]
            return self.features[step][feature_idx]
        except KeyError:
            raise KeyError(
                f"Feature '{feature_name}' not found. "
                f"Available features: {list(self.feature_map.keys())}"
            )

    def _get_current_price(self) -> float:
        """
        Get the current close price using the feature mapping.

        Returns:
        --------
        float
            Current closing price

        Raises:
        -------
        KeyError
            If 'Close' price column is not found in the data
        """
        try:
            close_idx = self.feature_map["Close"]
            return self.features[self.current_step][close_idx]
        except KeyError:
            raise KeyError(
                "'Close' price column not found in feature map. "
                f"Available features: {list(self.feature_map.keys())}"
            )

    def _create_feature_mapping(self, columns: List[str]) -> Dict[str, int]:
        """
        Create a mapping dictionary from feature names to their indices.

        Parameters:
        -----------
        columns : List[str]
            List of column names from the DataFrame

        Returns:
        --------
        Dict[str, int]
            Dictionary mapping feature names to their indices
        """
        # Create mapping of feature names to column indices
        feature_map = {col: idx for idx, col in enumerate(columns)}

        # Validate presence of required features
        if "Close" not in feature_map:
            raise ValueError(
                "Required 'Close' price column not found in data. "
                f"Available columns: {columns}"
            )

        return feature_map

    def _get_state(self, t: int) -> np.ndarray:
        """
        Get the state observation at time t.

        The state consists of price/indicator differences between consecutive
        time steps within the observation window.

        Parameters:
        -----------
        t : int
            Current time step

        Returns:
        --------
        np.ndarray
            Flattened array of feature differences in the observation window
        """
        n = self.window_size + 1
        d = t - n

        if d >= 0:
            # Get actual historical window
            window = self.features[d:t]
        else:
            # Pad with initial values for early steps
            window = np.array([self.features[0]] * n)

        # Calculate differences between consecutive time steps
        differences = sigmoid(window[1:] - window[:-1])

        # Flatten and return as 1D array
        return np.array([differences]).flatten()

    def format_price(self, price: float) -> str:
        """Format price value to currency string."""
        return f"${price:,.2f}"


if __name__ == "__main__":
    # Test script for the environment
    from src.preprocess import load_dataset

    # Load and prepare test data
    train_df, test_df = load_dataset(
        data_path="datasets/AAPL_2009-2010_6m_raw_1d.csv",
        is_auto=False,
        indicator_params={
            "ma_windows": [5, 20],
            "bb_window": 20,
            "bb_std": 2,
            "vol_window": 20,
        },
        verbose=True,
    )
    train_df = train_df[["Close", "BB_upper", "BB_lower"]]
    print("Testing Trading Environment...")

    # Initialize and test environment
    env = Environment(train_df, window_size=1, verbose=False)

    # Test reset functionality
    initial_state = env.reset()
    assert len(initial_state) == env.state_size, "Initial state size mismatch"
    assert env.current_step == 0, "Reset should set current_step to 0"
    print("✓ Reset test passed")

    # Test buy action
    next_state, reward, done, info = env.step(1)  # Buy
    assert len(next_state) == env.state_size, "Next state size mismatch"
    assert len(env.inventory) == 1, "Should have 1 position after buy"
    print("✓ Buy action test passed")

    # Test sell action
    next_state, reward, done, info = env.step(2)  # Sell
    assert len(env.inventory) == 0, "Should have 0 positions after sell"
    assert isinstance(reward, float), "Reward should be float"
    print("✓ Sell action test passed")

    # Test episode completion
    while not done:
        next_state, reward, done, info = env.step(0)  # Hold
    assert done, "Episode should end when reaching end of data"
    print("Final episode info:", info)
    print("✓ Episode completion test passed")
