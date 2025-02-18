import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis


def detect_outliers(data):
    """Detect if there are outliers in the data using IQR method."""
    Q1 = np.percentile(data, 25)
    Q3 = np.percentile(data, 75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return np.any((data < lower_bound) | (data > upper_bound))


def analyze_distribution(data):
    """Analyze the distribution of data and return appropriate normalization method and parameters."""
    if len(data) < 2:
        raise ValueError("Data must contain at least two values.")

    outliers_exist = detect_outliers(data)
    data_skewness = skew(data)
    data_kurtosis = kurtosis(data)

    if outliers_exist or abs(data_skewness) > 1 or data_kurtosis > 3:
        return "z-score", (np.mean(data), np.std(data))
    else:
        return "min-max", (np.min(data), np.max(data))


def choose_normalization(data):
    """
    Choose appropriate normalization method based on data distribution.

    Parameters:
    -----------
    data : array-like, pd.DataFrame
        Input data that can be 1D array, 2D array, or pandas DataFrame

    Returns:
    --------
    tuple : (methods, parameters)
        - For 1D array: (str, tuple) indicating method and parameters
        - For 2D array: (list, list) of methods and parameters for each column
        - For DataFrame: (dict, dict) of methods and parameters for each column
    """
    if isinstance(data, pd.DataFrame):
        results = {col: analyze_distribution(data[col].values) for col in data.columns}
        methods = {col: result[0] for col, result in results.items()}
        params = {col: result[1] for col, result in results.items()}
        return methods, params

    data_array = np.array(data)

    if data_array.ndim == 2:
        results = [
            analyze_distribution(data_array[:, i]) for i in range(data_array.shape[1])
        ]
        methods = [result[0] for result in results]
        params = [result[1] for result in results]
        return methods, params

    return analyze_distribution(data_array)


def normalize_data(data, method, params):
    """
    Normalize data using the specified method and parameters.

    Parameters:
    -----------
    data : array-like
        Data to normalize
    method : str
        'z-score' or 'min-max'
    params : tuple
        Parameters for normalization (mean, std) for z-score or (min, max) for min-max

    Returns:
    --------
    array-like
        Normalized data in the same format as input
    """
    if method == "z-score":
        mean, std = params
        return (data - mean) / std
    elif method == "min-max":
        min_val, max_val = params
        return (data - min_val) / (max_val - min_val)
    else:
        raise ValueError(f"Unknown normalization method: {method}")


def rolling_normalize(data, window_size, verbose=True):
    """
    Normalize data using a rolling window approach with automatic method selection.

    Parameters:
    -----------
    data : array-like
        Data to normalize
    window_size : int
        Size of the rolling window
    verbose : bool, optional (default=True)
        If True, returns both normalized data and window parameters
        If False, returns only normalized data

    Returns:
    --------
    array-like or tuple
        If verbose=True: (normalized_data, window_params)
            - normalized_data: Rolling normalized data
            - window_params: List of (method, params) tuples for each window
        If verbose=False: normalized_data only
    """
    data_array = np.array(data)
    normalized_data = np.zeros_like(data_array, dtype=float)
    window_params = []

    for i in range(len(data_array) - window_size + 1):
        window = data_array[i : i + window_size]
        method, params = choose_normalization(window)
        normalized_data[i : i + window_size] = normalize_data(window, method, params)
        if verbose:
            window_params.append((method, params))

    return (normalized_data, window_params) if verbose else normalized_data


def normalize_dataset(data, rolling=False, window_size=None):
    """
    Normalize entire dataset using either standard or rolling normalization.

    Parameters:
    -----------
    data : array-like, pd.DataFrame
        Input data to normalize
    rolling : bool
        Whether to use rolling normalization
    window_size : int
        Size of rolling window (required if rolling=True)

    Returns:
    --------
    normalized_data : same type as input
        Normalized version of input data
    methods : str, list, or dict
        Normalization methods used
    params : tuple, list, or dict
        Parameters used for normalization
    """
    if rolling and window_size is None:
        raise ValueError(
            "window_size must be specified when using rolling normalization"
        )

    # Handle DataFrame
    if isinstance(data, pd.DataFrame):
        normalized_data = pd.DataFrame(index=data.index)

        for col in data.columns:
            if rolling:
                norm_col = rolling_normalize(
                    data[col].values, window_size, verbose=False
                )
                normalized_data[col] = norm_col
            else:
                method, param = choose_normalization(data[col].values)
                normalized_data[col] = normalize_data(data[col].values, method, param)

        return normalized_data

    # Handle numpy array
    data_array = np.array(data)

    # Handle 2D array
    if len(data_array.shape) == 2:
        normalized_data = np.zeros_like(data_array, dtype=float)

        for i in range(data_array.shape[1]):
            if rolling:
                norm_col = rolling_normalize(
                    data_array[:, i], window_size, verbose=False
                )
                normalized_data[:, i] = norm_col
            else:
                method, param = choose_normalization(data_array[:, i])
                normalized_data[:, i] = normalize_data(data_array[:, i], method, param)

        return normalized_data

    # Handle 1D array
    if rolling:
        return rolling_normalize(data_array, window_size, verbose=False)
    else:
        method, param = choose_normalization(data_array)
        return normalize_data(data_array, method, param)


if __name__ == "__main__":
    from preprocess import process_stock_data
    from plots import plot_normalized_results

    # Load sample time series
    config = {
        "is_auto": False,
        "params": {
            "ma_windows": [5, 20],
            "bb_window": 20,
            "bb_std": 2,
            "vol_window": 20,
        },
        "data_path": "../dev/datasets/AAPL_2009-2010_6m_raw_1d.csv",
        "verbose": False,
    }

    # Preprocess
    sample_data = pd.read_csv(config["data_path"])
    processed_data = process_stock_data(
        sample_data,
        compute_indicator=True,
        is_auto=config["is_auto"],
        indicator_params=config["params"] if not config["is_auto"] else {},
        verbose=config["verbose"],
    )

    # Normalize dataset
    # regular normalize
    norm_regular = normalize_dataset(processed_data, rolling=False)

    # rolling nomalize
    window_size = 30
    norm_rolling = normalize_dataset(
        processed_data, rolling=True, window_size=window_size
    )

    # Show all plots
    plot_normalized_results(
        original=processed_data,
        norm_regular=norm_regular,
        norm_rolling=norm_rolling,
        window_size=window_size,
    )
