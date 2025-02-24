"""
Stock Data Processing Module
============================

This module provides comprehensive functionality for processing financial time series data
and calculating technical indicators with automatic parameter selection based on data
characteristics.

Key Features:
------------
1. Data Analysis and Preprocessing:
   - Missing value detection and handling
   - Data quality assessment
   - Automatic data cleaning
   - Time series validation


Usage Example:
-------------
- Basic usage with automatic parameter selection
```
processed_data = process_stock_data(
    data,
    compute_indicator=True,
    is_auto=True,
    verbose=True
)
```

- Manual parameter specification
```
processed_data = process_stock_data(
    data,
    compute_indicator=True,
    is_auto=False,
    indicator_params={
        'ma_windows': [5, 20],
        'bb_window': 20,
        'bb_std': 2,
        'vol_window': 20
    },
    verbose=True
)
```

Author: Tim Lin
Organization: DeepBioLab
License: MIT License
"""

import pandas as pd
from typing import Tuple

from src.features import calculate_technical_indicators


def analyze_stock_data(
    df: pd.DataFrame, verbose: bool = True
) -> Tuple[dict, pd.DataFrame]:
    """
    Analyze stock data for missing values and data quality issues.

    Parameters:
    -----------
    df : pd.DataFrame
        Input DataFrame containing stock data
    verbose : bool
        Whether to print analysis results

    Returns:
    --------
    dict
        Analysis results summary
    pd.DataFrame
        DataFrame with analysis statistics
    """
    analysis = {
        "total_rows": len(df),
        "missing_values": df.isnull().sum().to_dict(),
        "missing_percentage": (df.isnull().sum() / len(df) * 100).to_dict(),
    }
    analysis_df = pd.DataFrame(
        {
            "Missing Values": analysis["missing_values"],
            "Missing Percentage": analysis["missing_percentage"],
        }
    )

    if verbose:
        print("Data Analysis Summary")
        print(f"Total rows: {analysis['total_rows']}\n")
        print("Missing Values Analysis:")
        print(analysis_df)


def process_stock_data(
    df: pd.DataFrame,
    date_column: str = "Date",
    required_columns: list = None,
    compute_indicator: bool = False,
    is_auto: bool = True,
    indicator_params: dict = {},
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Process stock data by handling missing values, calculating technical indicators,
    and preparing it for further analysis.

    Parameters:
    -----------
    df : pd.DataFrame
        Input DataFrame containing stock data
    date_column : str
        Name of the date column
    required_columns : list
        List of required columns (default: ['Open', 'High', 'Low', 'Close', 'Volume'])
    calculate_features : bool
        Whether to calculate technical indicators
    verbose : bool
        Whether to print processing information

    Returns:
    --------
    pd.DataFrame
        Processed DataFrame with technical indicators
    """
    if verbose:
        print("Stock Data Processing Pipeline")

    if required_columns is None:
        required_columns = ["Open", "High", "Low", "Close", "Volume"]

    # Make a copy of the DataFrame
    data = df.copy()

    # Initial data check
    if verbose:
        print("1. Initial Analysis")
        analyze_stock_data(data, verbose=verbose)

    # Check required columns
    missing_columns = [col for col in required_columns if col not in data.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    # Set date as index
    if date_column in data.columns:
        data[date_column] = pd.to_datetime(data[date_column])
        data.set_index(date_column, inplace=True)

    # Handle missing values
    if verbose:
        print("2. Handling Missing Values By Forward Filling")

    data.ffill(inplace=True)

    # Calculate technical indicators
    if compute_indicator:
        if verbose:
            print("3. Technical Indicators")

        data = calculate_technical_indicators(
            data, **indicator_params, auto_select=is_auto, verbose=verbose
        )

        # Remove rows with NaN values from calculated features
        initial_rows = len(data)
        data.dropna(inplace=True)
        rows_dropped = initial_rows - len(data)

        if verbose and rows_dropped > 0:
            print(
                f"Dropped {rows_dropped} initial rows due to rolling window calculations"
            )

    # 6. Final check
    if verbose:
        print("4. Final Check")
        analyze_stock_data(data)

    return data


def load_dataset(data_path, indicator_params=None, verbose=False, is_auto=False):
    """
    Load and process stock data, then split into training and test sets.
    
    Parameters:
    -----------
    data_path : str
        Path to the stock data CSV file
    indicator_params : dict, optional
        Parameters for technical indicators calculation
    verbose : bool, default=False
        Whether to print processing information
    is_auto : bool, default=False
        Whether to use automatic parameter selection for indicators
        
    Returns:
    --------
    tuple
        (X_train, X_test) containing processed feature data split into
        training (80%) and test (20%) sets
    """
    data = pd.read_csv(data_path)
    processed_data = process_stock_data(
        data,
        compute_indicator=True,
        is_auto=is_auto,
        indicator_params=indicator_params,
        verbose=verbose
    )

    key_features = ['Close', 'Volume']
    additional_features = list(set(processed_data.columns).difference(set(data.columns)))
    feature_columns = key_features + additional_features
    data = processed_data[feature_columns]
    
    # Calculate split point for 80/20 train/test split
    split_idx = int(len(data) * 0.8)
    
    # Split the data
    train_data = data[:split_idx]
    test_data = data[split_idx:]
    
    if verbose:
        print(f"Training set size: {len(train_data)} samples")
        print(f"Test set size: {len(test_data)} samples")
    
    return train_data, test_data


# Example usage
if __name__ == "__main__":
    from plots import plot_technical_indicators

    # Configuration
    config = {
        "is_auto": False,
        "params": {
            "ma_windows": [5, 20],
            "bb_window": 20,
            "bb_std": 2,
            "vol_window": 20,
        },
        "data_path": "../datasets/AAPL_2009-2010_6m_raw_1d.csv",
        "verbose": False,
    }

    # Process and plot
    sample_data = pd.read_csv(config["data_path"])
    processed_data = process_stock_data(
        sample_data,
        compute_indicator=True,
        is_auto=config["is_auto"],
        indicator_params=config["params"] if not config["is_auto"] else {},
        verbose=config["verbose"],
    )
    plot_technical_indicators(processed_data)