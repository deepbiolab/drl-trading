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
    # Handle pandas DataFrame
    if isinstance(data, pd.DataFrame):
        results = {col: analyze_distribution(data[col].values) 
                  for col in data.columns}
        methods = {col: result[0] for col, result in results.items()}
        params = {col: result[1] for col, result in results.items()}
        return methods, params
    
    # Convert to numpy array if not already
    data_array = np.array(data)
    
    # Handle 2D array
    if len(data_array.shape) == 2:
        results = [analyze_distribution(data_array[:, i]) 
                  for i in range(data_array.shape[1])]
        methods = [result[0] for result in results]
        params = [result[1] for result in results]
        return methods, params
    
    # Handle 1D array
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

def normalize_dataset(data):
    """
    Normalize entire dataset (1D array, 2D array, or DataFrame) using appropriate methods.
    
    Parameters:
    -----------
    data : array-like, pd.DataFrame
        Input data to normalize
    
    Returns:
    --------
    normalized_data : same type as input
        Normalized version of input data
    methods : str, list, or dict
        Normalization methods used
    params : tuple, list, or dict
        Parameters used for normalization
    """
    methods, params = choose_normalization(data)
    
    # Handle DataFrame
    if isinstance(data, pd.DataFrame):
        normalized_data = pd.DataFrame(index=data.index)
        for col in data.columns:
            normalized_data[col] = normalize_data(data[col].values, 
                                               methods[col], 
                                               params[col])
        return normalized_data, methods, params
    
    # Handle numpy array
    data_array = np.array(data)
    
    # Handle 2D array
    if len(data_array.shape) == 2:
        normalized_data = np.zeros_like(data_array, dtype=float)
        for i in range(data_array.shape[1]):
            normalized_data[:, i] = normalize_data(data_array[:, i], 
                                                methods[i], 
                                                params[i])
        return normalized_data, methods, params
    
    # Handle 1D array
    return normalize_data(data_array, methods, params), methods, params

if __name__ == "__main__":
    # 1D array example
    print("1D Array Example:")
    data_1d = np.array([1, 2, 2, 3, 4, 100])  # Contains an outlier
    normalized_1d, method_1d, params_1d = normalize_dataset(data_1d)
    print("Original data:", data_1d)
    print("Normalized data:", normalized_1d)
    print(f"Method used: {method_1d}")
    print(f"Parameters: {params_1d}\n")

    # 2D array example
    print("2D Array Example:")
    data_2d = np.array([[1, 10, 100], 
                        [2, 20, 200], 
                        [3, 30, 300], 
                        [4, 40, 1000]])
    normalized_2d, methods_2d, params_2d = normalize_dataset(data_2d)
    print("Original data:\n", data_2d)
    print("Normalized data:\n", normalized_2d)
    print("Methods used:", methods_2d)
    print("Parameters:", params_2d, "\n")

    # DataFrame example
    print("DataFrame Example:")
    df = pd.DataFrame({
        'A': [1, 2, 3, 100],  # Contains outlier
        'B': [10, 20, 30, 40],  # Normal distribution
        'C': [1, 1, 1000, 1000]  # Skewed distribution
    })
    normalized_df, methods_df, params_df = normalize_dataset(df)
    print("Original DataFrame:\n", df)
    print("\nNormalized DataFrame:\n", normalized_df)
    print("\nMethods used:", methods_df)
    print("Parameters:", params_df)