"""
Stock Data Processing and Technical Analysis Tool
==============================================

This module provides functionality for processing stock market data and calculating
various technical indicators including Moving Averages, Bollinger Bands, and Volatility.
It also includes visualization tools for technical analysis.

Features:
- Data preprocessing and cleaning
- Automatic/Manual technical indicator parameter selection
- Technical indicator calculation
- Visualization of technical indicators

Author: Tim Lin
Organization: DeepBioLab
License: MIT License

Copyright (c) 2024 Tim Lin - DeepBioLab

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Tuple


# Add these utility functions at the top
def print_section(title: str) -> None:
    """Print a section header."""
    print(f"\n{'='*80}\n{title.center(80)}\n{'='*80}")


def print_subsection(title: str) -> None:
    """Print a subsection header."""
    print(f"\n=== {title} ===")


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

    return analysis, analysis_df


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
    if verbose:
        print_section("Technical Indicator Calculation")
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
            print_subsection("Selected Parameters")
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
        print_subsection("Calculating Indicators")

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
        print_section("Stock Data Processing Pipeline")

    if required_columns is None:
        required_columns = ["Open", "High", "Low", "Close", "Volume"]

    # Make a copy of the DataFrame
    data = df.copy()

    # Initial data check
    if verbose:
        print_subsection("1. Initial Analysis")

    initial_analysis, analysis_df = analyze_stock_data(data, verbose=verbose)

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
        print_subsection("2. Handling Missing Values By Forward Filling")

    data.ffill(inplace=True)

    # Calculate technical indicators
    if compute_indicator:
        if verbose:
            print_subsection("3. Technical Indicators")
        
        data = calculate_technical_indicators(data, **indicator_params, auto_select=is_auto, verbose=verbose)

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
        print_subsection("4. Final Check")
        final_analysis, final_analysis_df = analyze_stock_data(data)

    return data


def plot_technical_indicators(
    data: pd.DataFrame,
    price_col: str = "Close",
    ma_windows: list = None,
    bb_window: int = None,
    vol_window: int = None,
    additional_indicators: list = None,
    save_path: str = None,
    figsize: tuple = (15, 20),
) -> None:
    """
    Plot technical indicators for visual analysis.

    Parameters:
    -----------
    data : pd.DataFrame
        Processed DataFrame with technical indicators
    price_col : str
        Name of the column containing price data
    ma_windows : list
        List of windows for moving averages to plot (if None, auto-detect from data)
    bb_window : int
        Window used for Bollinger Bands (if None, auto-detect from data)
    vol_window : int
        Window used for volatility calculation (if None, auto-detect from data)
    additional_indicators : list
        List of additional indicators to plot
    save_path : str
        Path to save the plots (optional)
    figsize : tuple
        Figure size for the plots
    """
    print_subsection("Generating Plots")
    # Auto-detect parameters if not provided
    if ma_windows is None:
        ma_windows = sorted(
            [int(col[2:]) for col in data.columns if col.startswith("MA")]
        )

    if bb_window is None:
        bb_cols = [col for col in data.columns if col.startswith("STD")]
        if bb_cols:
            bb_window = int(bb_cols[0][3:])

    if vol_window is None:
        vol_cols = [
            col for col in data.columns if col.startswith("Vol") and col != "Volume"
        ]
        if vol_cols:
            vol_window = int(vol_cols[0][3:])

    # Determine number of subplots based on available indicators
    n_plots = 3  # Base plots: BB, MA, Volatility
    if additional_indicators:
        n_plots += len(additional_indicators)

    plt.style.use("ggplot")
    fig, axes = plt.subplots(n_plots, 1, figsize=figsize)

    plot_idx = 0

    # Plot 1: Price with Bollinger Bands
    if bb_window and all(
        col in data.columns for col in [f"MA{bb_window}", "BB_upper", "BB_lower"]
    ):
        ax1 = axes[plot_idx]
        ax1.plot(data.index, data[price_col], label=price_col, color="blue", alpha=0.5)
        ax1.plot(
            data.index,
            data[f"MA{bb_window}"],
            label=f"MA{bb_window}",
            color="red",
            alpha=0.5,
        )
        ax1.plot(
            data.index,
            data["BB_upper"],
            label="BB Upper",
            color="black",
            alpha=0.5,
            linestyle="--",
        )
        ax1.plot(
            data.index,
            data["BB_lower"],
            label="BB Lower",
            color="black",
            alpha=0.5,
            linestyle="--",
        )
        ax1.set_title(f"Price with Bollinger Bands ({bb_window}-day)")
        ax1.set_xlabel("Date")
        ax1.set_ylabel("Price")
        ax1.legend()
        ax1.grid(True)
        plot_idx += 1

    # Plot 2: Moving Averages
    if ma_windows:
        ax2 = axes[plot_idx]
        ax2.plot(data.index, data[price_col], label=price_col, color="blue", alpha=0.5)
        colors = [
            "red",
            "green",
            "purple",
            "orange",
            "brown",
            "pink",
        ]  # Add more colors if needed
        for ma_window, color in zip(ma_windows, colors):
            if f"MA{ma_window}" in data.columns:
                ax2.plot(
                    data.index,
                    data[f"MA{ma_window}"],
                    label=f"MA{ma_window}",
                    color=color,
                    alpha=0.5,
                )
        ax2.set_title("Price with Moving Averages")
        ax2.set_xlabel("Date")
        ax2.set_ylabel("Price")
        ax2.legend()
        ax2.grid(True)
        plot_idx += 1

    # Plot 3: Volatility
    if vol_window and f"Vol{vol_window}" in data.columns:
        ax3 = axes[plot_idx]
        ax3.plot(
            data.index,
            data[f"Vol{vol_window}"],
            label=f"{vol_window}-day Volatility",
            color="blue",
            alpha=0.5,
        )
        ax3.set_title(f"Historical Volatility ({vol_window}-day)")
        ax3.set_xlabel("Date")
        ax3.set_ylabel("Volatility")
        ax3.legend()
        ax3.grid(True)
        plot_idx += 1

    # Plot additional indicators
    if additional_indicators:
        for indicator in additional_indicators:
            if indicator == "momentum" and "Momentum14" in data.columns:
                ax = axes[plot_idx]
                ax.plot(
                    data.index,
                    data["Momentum14"],
                    label="14-day Momentum",
                    color="purple",
                    alpha=0.5,
                )
                ax.set_title("14-day Momentum")
                ax.set_xlabel("Date")
                ax.set_ylabel("Momentum")
                ax.legend()
                ax.grid(True)
                plot_idx += 1

            elif indicator == "rsi" and "RSI14" in data.columns:
                ax = axes[plot_idx]
                ax.plot(
                    data.index,
                    data["RSI14"],
                    label="14-day RSI",
                    color="orange",
                    alpha=0.5,
                )
                # Add RSI overbought/oversold lines
                ax.axhline(y=70, color="r", linestyle="--", alpha=0.5)
                ax.axhline(y=30, color="g", linestyle="--", alpha=0.5)
                ax.set_title("Relative Strength Index (RSI)")
                ax.set_xlabel("Date")
                ax.set_ylabel("RSI")
                ax.legend()
                ax.grid(True)
                plot_idx += 1

    # Adjust x-axis ticks for all plots
    for ax in axes:
        ax.tick_params(axis="x", rotation=45)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
    plt.show()


# Example usage
if __name__ == "__main__":
    # Configuration
    config = {
        'is_auto': False,
        'params': {
            'ma_windows': [5, 20],
            'bb_window': 20,
            'bb_std': 2,
            'vol_window': 20
        },
        'data_path': "../dev/datasets/AAPL_2009-2010_6m_raw_1d.csv",
        'verbose': False,
    }
    
    # Process and plot
    sample_data = pd.read_csv(config['data_path'])
    processed_data = process_stock_data(
        sample_data, 
        compute_indicator=True,
        is_auto=config['is_auto'],
        indicator_params=config['params'] if not config['is_auto'] else {},
        verbose=config['verbose']
    )
    plot_technical_indicators(processed_data)