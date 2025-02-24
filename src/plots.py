import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots


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
    print("Generating Plots")
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


def plot_normalized_results(
    original, norm_regular=None, norm_rolling=None, window_size=None, key_metrics=None
):
    """
    Plot comparison of original and normalized data using Plotly.
    Adaptively shows plots based on provided normalization results.

    Parameters:
    -----------
    original : pd.DataFrame
        Original data
    norm_regular : pd.DataFrame, optional
        Regular normalized data
    norm_rolling : pd.DataFrame, optional
        Rolling normalized data
    window_size : int, optional
        Size of rolling window used (required if norm_rolling is provided)
    key_metrics : list, optional
        List of column names to show by default. If None, uses ['Close', 'Volume']
    """

    # Default key metrics if not specified
    if key_metrics is None:
        key_metrics = (
            ["Close", "Volume"]
            if all(metric in original.columns for metric in ["Close", "Volume"])
            else original.columns[:2]
        )

    # Determine which plots to show
    plots_to_show = ["original"]
    if norm_regular is not None:
        plots_to_show.append("regular")
    if norm_rolling is not None:
        if window_size is None:
            raise ValueError(
                "window_size must be provided when plotting rolling normalization"
            )
        plots_to_show.append("rolling")

    num_plots = len(plots_to_show)

    # Create subplot titles
    subplot_titles = []
    for plot_type in plots_to_show:
        if plot_type == "original":
            subplot_titles.append("Original Time Series")
        elif plot_type == "regular":
            subplot_titles.append("Regular Normalization")
        else:  # rolling
            subplot_titles.append(f"Rolling Normalization (window={window_size})")

    # Create subplots
    fig = make_subplots(
        rows=num_plots, cols=1, subplot_titles=subplot_titles, vertical_spacing=0.1
    )

    # Add traces for each column in the data
    for column in original.columns:
        plot_row = 1
        # Determine if this column should be visible by default
        visible = column in key_metrics

        # Original data
        if "original" in plots_to_show:
            fig.add_trace(
                go.Scatter(
                    x=original.index,
                    y=original[column],
                    name=f"Original - {column}",
                    legendgroup="original",
                    legendgrouptitle_text="Original Data",
                    visible=(
                        True if visible else "legendonly"
                    ),  # Show only key metrics by default
                ),
                row=plot_row,
                col=1,
            )
            plot_row += 1

        # Regular normalized data
        if "regular" in plots_to_show:
            fig.add_trace(
                go.Scatter(
                    x=norm_regular.index,
                    y=norm_regular[column],
                    name=f"Regular - {column}",
                    legendgroup="regular",
                    legendgrouptitle_text="Regular Normalization",
                    visible=(
                        True if visible else "legendonly"
                    ),  # Show only key metrics by default
                ),
                row=plot_row,
                col=1,
            )
            plot_row += 1

        # Rolling normalized data
        if "rolling" in plots_to_show:
            fig.add_trace(
                go.Scatter(
                    x=norm_rolling.index,
                    y=norm_rolling[column],
                    name=f"Rolling - {column}",
                    legendgroup="rolling",
                    legendgrouptitle_text="Rolling Normalization",
                    visible=(
                        True if visible else "legendonly"
                    ),  # Show only key metrics by default
                ),
                row=plot_row,
                col=1,
            )

    # Calculate appropriate height based on number of plots
    height_per_plot = 300  # pixels per plot
    total_height = max(height_per_plot * num_plots, 400)  # minimum height of 400px

    # Update layout
    fig.update_layout(
        height=total_height,
        showlegend=True,
        legend=dict(groupclick="toggleitem", itemclick="toggle", title="Metrics"),
        title_text="Normalization Comparison",
        title_x=0.5,
    )

    # Update axes labels
    fig.update_xaxes(title_text="Time")
    fig.update_yaxes(title_text="Value")

    # Add gridlines
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor="LightGray")
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor="LightGray")

    # Show the plot
    fig.show()


def plot_behavior(env, states_buy, states_sell, total_reward, train=True):
    """
    Plot trading behavior with price data and buy/sell signals

    Parameters:
    -----------
    env : TradingEnv
        Trading environment containing the price data and indicators
    states_buy : list
        List of indices where buy actions occurred
    states_sell : list
        List of indices where sell actions occurred
    total_reward : float
        Total reward/profit from the episode
    train : bool
        Whether this is training or test data (affects x-axis ticks)
    """
    fig = plt.figure(figsize=(15, 5))

    # Get data from environment
    close_data = env.data["Close"].values
    bb_upper_data = env.data["BB_upper"].values
    bb_lower_data = env.data["BB_lower"].values

    # Plot price and Bollinger Bands
    plt.plot(close_data, color="k", lw=2.0, label="Close Price")
    plt.plot(bb_upper_data, color="b", lw=2.0, label="Bollinger Bands")
    plt.plot(bb_lower_data, color="b", lw=2.0)

    # Plot buy/sell signals
    if states_buy:
        plt.plot(
            states_buy,
            close_data[states_buy],
            "^",
            markersize=10,
            color="g",
            label="Buy Signal",
        )
    if states_sell:
        plt.plot(
            states_sell,
            close_data[states_sell],
            "v",
            markersize=10,
            color="r",
            label="Sell Signal",
        )

    # Set title and labels
    plt.title(f"Trading Behavior (Total Reward: ${total_reward:.2f})")
    plt.xlabel("Time Step")
    plt.ylabel("Price ($)")
    plt.legend()

    # Set x-axis ticks based on data length
    data_len = len(close_data)
    if train:
        tick_interval = max(1, data_len // 15)
    else:
        tick_interval = max(1, data_len // 2)

    plt.xticks(range(0, data_len, tick_interval), rotation=45)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_losses(losses, title):
    """Plot the training losses"""
    plt.figure(figsize=(10, 6))
    plt.plot(losses, label="Training Loss")
    plt.title(title)
    plt.xlabel("Episode")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.show()
