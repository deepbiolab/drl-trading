"""
Data Normalization Module with Static and Rolling Normalization
==============================================

This module provides functionality for data normalization with automatic method selection
and verification capabilities. It implements both static and rolling window normalization approaches
with built-in state tracking and inverse transformation support.

Key Features:
------------
1. Automatic Normalization Method Selection:
   - Analyzes data distribution characteristics
   - Chooses between z-score and min-max normalization based on:
     * Presence of outliers (IQR method)
     * Skewness
     * Kurtosis

2. Normalization Approaches:
   - Static normalization (whole dataset)
   - Rolling window normalization
   - Support for both manual and automatic method selection

3. Data Processing:
   - Handles pandas DataFrames
   - Automatic detection of numeric columns
   - Preserves non-numeric data

4. State Management:
   - Tracks normalization methods per column
   - Maintains transformation states for inverse operations
   - Stores scalers for both static and rolling approaches

5. Quality Assurance:
   - Built-in verification functionality
   - MSE-based accuracy checking
   - Detailed verification reporting

Usage Example:
-------------
- Static normalization

`normalized_data, normalizer = normalize_dataset(data, rolling=False, is_auto=True)`

- Rolling window normalization

`normalized_data, normalizer = normalize_dataset(data, rolling=True, window_size=30, is_auto=True)`

- Verification

`results = verify_normalization(original_data, normalized_data, normalizer)`

Author: Tim Lin
Organization: DeepBioLab
License: MIT License
"""

import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from typing import Dict, Tuple, Union, List, Optional


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


class DataNormalizer:
    """
    A class to handle both static and rolling normalization of financial data
    with state space tracking and scikit-learn integration.
    """

    def __init__(self):
        self.static_scalers: Dict[str, Union[StandardScaler, MinMaxScaler]] = {}
        self.rolling_scalers: Dict[str, List[Union[StandardScaler, MinMaxScaler]]] = {}
        self.numeric_columns: List[str] = []
        self.normalization_methods: Dict[str, str] = {}  # Track method per column

    def _is_numeric_column(self, series: pd.Series) -> bool:
        """Check if a column is numeric and should be normalized."""
        return pd.api.types.is_numeric_dtype(series) and not series.isnull().all()

    def _get_numeric_columns(self, data: pd.DataFrame) -> List[str]:
        """Identify numeric columns for normalization."""
        return [col for col in data.columns if self._is_numeric_column(data[col])]

    def _initialize_scalers(
        self, column: str, data: np.ndarray, is_auto: bool = True
    ) -> None:
        """
        Initialize appropriate scaler for a column based on data distribution if auto,
        otherwise use specified scaler type.
        """
        if is_auto:
            method, _ = analyze_distribution(data)
            self.normalization_methods[column] = method
            self.static_scalers[column] = (
                MinMaxScaler() if method == "min-max" else StandardScaler()
            )
        else:
            self.static_scalers[column] = (
                StandardScaler()
            )  # Default to StandardScaler when not auto

    def static_normalize(
        self, data: pd.DataFrame, is_auto: bool = True
    ) -> pd.DataFrame:
        """
        Perform static normalization on the entire dataset.

        Parameters:
        -----------
        data : pd.DataFrame
            Input data to normalize
        is_auto : bool
            If True, automatically select normalization method based on data distribution

        Returns:
        --------
        pd.DataFrame
            Normalized dataset
        """
        normalized_data = data.copy()
        self.numeric_columns = self._get_numeric_columns(data)

        for column in self.numeric_columns:
            values = data[column].values.reshape(-1, 1)
            self._initialize_scalers(column, values.flatten(), is_auto)
            normalized_data[column] = self.static_scalers[column].fit_transform(values)

        return normalized_data

    def rolling_normalize(
        self, data: pd.DataFrame, window_size: int, is_auto: bool = True
    ) -> pd.DataFrame:
        """
        Perform rolling window normalization.

        Parameters:
        -----------
        data : pd.DataFrame
            Input data to normalize
        window_size : int
            Size of the rolling window
        is_auto : bool
            If True, automatically select normalization method based on data distribution

        Returns:
        --------
        pd.DataFrame
            Rolling normalized dataset
        """
        normalized_data = data.copy()
        self.numeric_columns = self._get_numeric_columns(data)

        for column in self.numeric_columns:
            self.rolling_scalers[column] = []
            values = data[column].values
            normalized_values = np.zeros_like(values, dtype=float)

            for i in range(len(data) - window_size + 1):
                window = values[i : i + window_size].reshape(-1, 1)

                # Determine scaler type for this window
                if is_auto:
                    method, _ = analyze_distribution(window.flatten())
                    scaler = MinMaxScaler() if method == "min-max" else StandardScaler()
                else:
                    scaler = StandardScaler()  # Default when not auto

                scaler.fit(window)
                self.rolling_scalers[column].append(scaler)

                if i == 0:
                    normalized_values[:window_size] = scaler.transform(window).flatten()
                else:
                    last_value = values[i + window_size - 1].reshape(1, -1)
                    normalized_values[i + window_size - 1] = scaler.transform(
                        last_value
                    ).flatten()[0]

            normalized_data[column] = normalized_values

        return normalized_data

    def inverse_transform_static(self, data: pd.DataFrame) -> pd.DataFrame:
        """Reverse static normalization."""
        denormalized_data = data.copy()

        for column in self.numeric_columns:
            if column in self.static_scalers:
                values = data[column].values.reshape(-1, 1)
                denormalized_data[column] = self.static_scalers[
                    column
                ].inverse_transform(values)

        return denormalized_data

    def inverse_transform_rolling(
        self, data: pd.DataFrame, window_index: int, columns: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Reverse rolling normalization for a specific window.

        Parameters:
        -----------
        data : pd.DataFrame
            Normalized data
        window_index : int
            Index of the window to reverse
        columns : List[str], optional
            Specific columns to denormalize

        Returns:
        --------
        pd.DataFrame
            Denormalized data for the specified window
        """
        denormalized_data = data.copy()
        columns_to_process = columns if columns is not None else self.numeric_columns

        for column in columns_to_process:
            if column in self.rolling_scalers:
                if 0 <= window_index < len(self.rolling_scalers[column]):
                    scaler = self.rolling_scalers[column][window_index]
                    values = data[column].values.reshape(-1, 1)
                    denormalized_data[column] = scaler.inverse_transform(values)

        return denormalized_data


def normalize_dataset(
    data: pd.DataFrame,
    rolling: bool = False,
    window_size: Optional[int] = None,
    is_auto: bool = True,
) -> Union[pd.DataFrame, Tuple[pd.DataFrame, DataNormalizer]]:
    """
    Main function to normalize dataset using either static or rolling normalization.

    Parameters:
    -----------
    data : pd.DataFrame
        Input data to normalize
    rolling : bool
        Whether to use rolling normalization
    window_size : int, optional
        Size of rolling window (required if rolling=True)
    is_auto : bool
        If True, automatically select normalization method based on data distribution

    Returns:
    --------
    Union[pd.DataFrame, Tuple[pd.DataFrame, DataNormalizer]]
        Normalized data and optionally the normalizer object
    """
    normalizer = DataNormalizer()

    if rolling:
        if window_size is None:
            raise ValueError("window_size must be specified for rolling normalization")
        normalized_data = normalizer.rolling_normalize(data, window_size, is_auto)
    else:
        normalized_data = normalizer.static_normalize(data, is_auto)

    return normalized_data, normalizer


def verify_normalization(
    original_data: pd.DataFrame,
    normalized_data: pd.DataFrame,
    normalizer: DataNormalizer,
    rolling: bool = False,
    window_size: Optional[int] = None,
    tolerance: float = 1e-10,
) -> Dict[str, Dict[str, float]]:
    """
    Verify normalization accuracy by comparing original data with inverse transformed data.

    Parameters:
    -----------
    original_data : pd.DataFrame
        Original data before normalization
    normalized_data : pd.DataFrame
        Normalized data
    normalizer : DataNormalizer
        Normalizer object used for the transformation
    rolling : bool
        Whether rolling normalization was used
    window_size : int, optional
        Size of rolling window (required if rolling=True)
    tolerance : float
        Maximum acceptable MSE value for verification

    Returns:
    --------
    Dict[str, Dict[str, float]]
        Dictionary containing MSE values for each column
        Format: {column_name: {'mse': mse_value, 'passed': bool}}

    Raises:
    -------
    ValueError
        If MSE exceeds tolerance for any column
    """
    results = {}

    if rolling:
        if window_size is None:
            raise ValueError(
                "window_size must be specified for rolling normalization verification"
            )

        # Verify first window for rolling normalization
        window_idx = 0
        window_start = window_idx
        window_end = window_idx + window_size

        denorm_data = normalizer.inverse_transform_rolling(
            normalized_data.iloc[window_start:window_end], window_index=window_idx
        )

        for column in normalizer.numeric_columns:
            mse = np.mean(
                (
                    original_data[column].iloc[window_start:window_end]
                    - denorm_data[column]
                )
                ** 2
            )
            passed = mse <= tolerance
            results[column] = {"mse": mse, "passed": passed, "window_index": window_idx}

    else:
        # Verify static normalization
        denorm_data = normalizer.inverse_transform_static(normalized_data)

        for column in normalizer.numeric_columns:
            mse = np.mean((original_data[column] - denorm_data[column]) ** 2)
            passed = mse <= tolerance
            results[column] = {"mse": mse, "passed": passed}

    # Check if any verifications failed
    failed_columns = [col for col, res in results.items() if not res["passed"]]
    if failed_columns:
        raise ValueError(
            f"Normalization verification failed for columns: {failed_columns}\n"
            f"Results: {results}"
        )

    return results


if __name__ == "__main__":
    from preprocess import process_stock_data
    from plots import plot_normalized_results

    # ==== Load sample time series ====
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

    # ==== Preprocess ====
    sample_data = pd.read_csv(config["data_path"])
    processed_data = process_stock_data(
        sample_data,
        compute_indicator=True,
        is_auto=config["is_auto"],
        indicator_params=config["params"] if not config["is_auto"] else {},
        verbose=config["verbose"],
    )

    # ==== Normalize dataset ====
    # Regular normalize with automatic method selection
    norm_regular, static_normalizer = normalize_dataset(
        processed_data, rolling=False, is_auto=True
    )
    print("\nStatic Normalization Methods:")
    print(static_normalizer.normalization_methods)

    # Rolling normalize with automatic method selection
    window_size = 30
    norm_rolling, rolling_normalizer = normalize_dataset(
        processed_data, rolling=True, window_size=window_size, is_auto=True
    )

    # Show all plots
    plot_normalized_results(
        original=processed_data,
        norm_regular=norm_regular,
        norm_rolling=norm_rolling,
        window_size=window_size,
    )

    # Verify static normalization
    try:
        static_results = verify_normalization(
            processed_data, norm_regular, static_normalizer, rolling=False
        )
        print("\nStatic Normalization Verification Results:")
        for column, result in static_results.items():
            print(f"{column}: MSE = {result['mse']:.10f} (Passed: {result['passed']})")
    except ValueError as e:
        print(f"Static normalization verification failed: {e}")

    # Verify rolling normalization
    try:
        rolling_results = verify_normalization(
            processed_data,
            norm_rolling,
            rolling_normalizer,
            rolling=True,
            window_size=window_size,
        )
        print("\nRolling Normalization Verification Results:")
        for column, result in rolling_results.items():
            print(
                f"{column}: MSE = {result['mse']:.10f} "
                f"(Passed: {result['passed']}, Window: {result['window_index']})"
            )
    except ValueError as e:
        print(f"Rolling normalization verification failed: {e}")
