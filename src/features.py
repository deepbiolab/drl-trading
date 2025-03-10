"""
Stock Technical Indicators Computation Module
=============================================

This module calculating technical indicators with automatic parameter selection based on data
characteristics.

1. Technical Indicator Calculation:
   - Moving Averages (Multiple windows)
   - Bollinger Bands
   - Volatility Measures
   - Additional indicators based on data characteristics:
     * Momentum
     * RSI (Relative Strength Index)

2. Automatic Parameter Selection:
   - Data-driven parameter optimization
   - Adapts to:
     * Time series length
     * Volatility levels
     * Price characteristics
     * Market conditions
     
Author: Tim Lin
Organization: DeepBioLab
License: MIT License     
"""

import numpy as np
import pandas as pd


def analyze_price_characteristics(data: pd.DataFrame, price_col: str = "Close") -> dict:
    """
    Analyze price data characteristics to determine appropriate technical indicators.

    Parameters:
    -----------
    data : pd.DataFrame
        Input DataFrame containing price data
    price_col : str
        Name of the column containing price data

    Returns:
    --------
    dict
        Dictionary containing analysis results and recommended parameters
    """
    price_series = data[price_col]
    returns = np.log(price_series / price_series.shift(1)).dropna()

    # Calculate basic statistics
    volatility = returns.std() * np.sqrt(252)
    skewness = returns.skew()
    kurtosis = returns.kurtosis()

    # Calculate price trend characteristics
    total_days = len(data)
    price_range = price_series.max() - price_series.min()
    price_volatility = price_series.std() / price_series.mean()

    # Determine appropriate parameters based on data characteristics
    analysis = {
        "total_days": total_days,
        "volatility": volatility,
        "skewness": skewness,
        "kurtosis": kurtosis,
        "price_range_ratio": price_range / price_series.mean(),
        "price_volatility": price_volatility,
    }

    return analysis


def select_technical_indicators(analysis: dict) -> dict:
    """
    Select appropriate technical indicators based on data analysis.

    Parameters:
    -----------
    analysis : dict
        Dictionary containing analysis results

    Returns:
    --------
    dict
        Dictionary containing selected indicators and their parameters
    """
    total_days = analysis["total_days"]
    volatility = analysis["volatility"]
    price_volatility = analysis["price_volatility"]

    # Initialize parameters dictionary
    params = {
        "ma_windows": [],
        "bb_window": 20,
        "bb_std": 2,
        "vol_window": 20,
        "additional_indicators": [],
    }

    # Select moving average windows based on data length and volatility
    if total_days >= 252:  # More than a year of data
        params["ma_windows"] = [5, 20, 50, 200]
    elif total_days >= 126:  # More than 6 months
        params["ma_windows"] = [5, 20, 50]
    else:  # Less than 6 months
        params["ma_windows"] = [5, 20]

    # Adjust Bollinger Bands parameters based on volatility
    if volatility > 0.4:  # High volatility
        params["bb_std"] = 2.5
        params["bb_window"] = 15
    elif volatility < 0.15:  # Low volatility
        params["bb_std"] = 1.5
        params["bb_window"] = 25

    # Adjust volatility window based on price volatility
    if price_volatility > 0.2:
        params["vol_window"] = 15
    elif price_volatility < 0.05:
        params["vol_window"] = 30

    # Add additional indicators based on characteristics
    if analysis["skewness"] > 1 or analysis["skewness"] < -1:
        params["additional_indicators"].append("momentum")
    if analysis["kurtosis"] > 3:
        params["additional_indicators"].append("rsi")

    return params


def calculate_additional_indicators(
    df: pd.DataFrame, indicators: list, price_col: str = "Close"
) -> pd.DataFrame:
    """
    Calculate additional technical indicators based on data characteristics.

    Parameters:
    -----------
    df : pd.DataFrame
        Input DataFrame
    indicators : list
        List of additional indicators to calculate
    price_col : str
        Name of the price column

    Returns:
    --------
    pd.DataFrame
        DataFrame with additional indicators
    """
    data = df.copy()

    for indicator in indicators:
        if indicator == "momentum":
            # Calculate 14-day momentum
            data["Momentum14"] = data[price_col] - data[price_col].shift(14)

        elif indicator == "rsi":
            # Calculate 14-day RSI
            delta = data[price_col].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            data["RSI14"] = 100 - (100 / (1 + rs))

    return data


def calculate_technical_indicators(
    data: pd.DataFrame,
    price_col: str = "Close",
    auto_select: bool = True,
    ma_windows: list = None,
    bb_window: int = None,
    bb_std: int = None,
    vol_window: int = None,
    annualization_factor: int = 252,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Calculate technical indicators for financial time series data with automatic parameter selection.

    Parameters:
    -----------
    data : pd.DataFrame
        Input DataFrame containing price data
    price_col : str
        Name of the column containing price data
    auto_select : bool
        Whether to automatically select indicators based on data characteristics
    ma_windows : list
        List of windows for moving averages (optional if auto_select=True)
    bb_window : int
        Window for Bollinger Bands calculation (optional if auto_select=True)
    bb_std : int
        Number of standard deviations for Bollinger Bands (optional if auto_select=True)
    vol_window : int
        Window for volatility calculation (optional if auto_select=True)
    annualization_factor : int
        Factor for annualizing volatility
    verbose : bool
        Whether to print processing information

    Returns:
    --------
    pd.DataFrame
        DataFrame with added technical indicators
    """
    if price_col not in data.columns:
        raise ValueError(f"Column '{price_col}' not found in the dataset.")

    if verbose:
        print("Technical Indicator Calculation")
        mode = "AUTO" if auto_select else "MANUAL"
        print(f"Running in {mode} mode")

    df = data.copy()

    if auto_select:
        # Analyze data characteristics
        analysis = analyze_price_characteristics(df, price_col)

        # Select appropriate indicators
        params = select_technical_indicators(analysis)

        # Use selected parameters
        ma_windows = params["ma_windows"]
        bb_window = params["bb_window"]
        bb_std = params["bb_std"]
        vol_window = params["vol_window"]

        if verbose:
            print("Selected Parameters:")
            print(f"• MA Windows: {ma_windows}")
            print(f"• Bollinger Bands: {bb_window}-day, {bb_std} std")
            print(f"• Volatility: {vol_window}-day")
            if params["additional_indicators"]:
                print(f"• Additional: {params['additional_indicators']}")
    else:
        # Use default values if not provided
        ma_windows = ma_windows or [5, 20]
        bb_window = bb_window or 20
        bb_std = bb_std or 2
        vol_window = vol_window or 20

    if verbose:
        print("Calculating Indicators")

    # Calculate Moving Averages
    for window in ma_windows:
        df[f"MA{window}"] = df[price_col].rolling(window).mean()
        if verbose:
            print(f"✓ MA{window}")

    # Calculate Bollinger Bands
    df[f"STD{bb_window}"] = df[price_col].rolling(bb_window).std()
    df[f"BB_upper"] = df[f"MA{bb_window}"] + bb_std * df[f"STD{bb_window}"]
    df[f"BB_lower"] = df[f"MA{bb_window}"] - bb_std * df[f"STD{bb_window}"]
    if verbose:
        print(f"✓ Bollinger Bands ({bb_window}-day)")

    # Calculate Log Returns
    df["Log_Ret"] = np.log(df[price_col] / df[price_col].shift(1))

    # Calculate Volatility
    df[f"Vol{vol_window}"] = df["Log_Ret"].rolling(window=vol_window).std() * np.sqrt(
        annualization_factor
    )
    if verbose:
        print(f"✓ Volatility ({vol_window}-day)")

    # Calculate additional indicators if auto_select is True
    if auto_select and params["additional_indicators"]:
        df = calculate_additional_indicators(
            df, params["additional_indicators"], price_col
        )

    return df
