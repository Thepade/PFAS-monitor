import os
import pickle
from typing import Tuple, List

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

import config


def load_data(data_path: str) -> pd.DataFrame:
    """
    Load data from a CSV file and parse timestamps.
    
    Args:
        data_path (str): Path to the CSV file.
        
    Returns:
        pd.DataFrame: Loaded data with timestamp parsed as datetime.
    """
    df = pd.read_csv(data_path)
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the data by forward-filling then backward-filling missing values,
    and clipping outliers using the IQR rule.
    
    Args:
        df (pd.DataFrame): Input data.
        
    Returns:
        pd.DataFrame: Cleaned data.
    """
    df_clean = df.copy()
    
    # Forward-fill then backward-fill missing values
    df_clean = df_clean.ffill().bfill()
    
    # Clip outliers using IQR rule on numeric columns
    numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        q1 = df_clean[col].quantile(0.25)
        q3 = df_clean[col].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - config.IQR_MULTIPLIER * iqr
        upper_bound = q3 + config.IQR_MULTIPLIER * iqr
        
        # Clip values to bounds
        df_clean[col] = df_clean[col].clip(lower=lower_bound, upper=upper_bound)
        
    return df_clean


def calibrate_signal(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add a concentration_calibrated column based on the raw signal.
    
    Args:
        df (pd.DataFrame): Input data containing 'signal_mV'.
        
    Returns:
        pd.DataFrame: Data with 'concentration_calibrated' column added.
    """
    df_calibrated = df.copy()
    if 'signal_mV' in df_calibrated.columns:
        # concentration = A * signal + B
        df_calibrated['concentration_calibrated'] = (
            config.CALIBRATION_A * df_calibrated['signal_mV'] + config.CALIBRATION_B
        )
    return df_calibrated


def prepare_data(
    data_path: str, 
    include_env_features: bool = True,
    scaler_filename: str = 'scaler.pkl'
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, MinMaxScaler]:
    """
    Full pipeline: load -> clean -> calibrate -> normalize -> create sliding windows.
    Fits scaler ONLY on train set and saves it as scaler.pkl in the same directory as data file.
    
    Args:
        data_path (str): Path to the CSV data file.
        include_env_features (bool, optional): Whether to include environmental features. Defaults to True.
        
    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, MinMaxScaler]: 
            X_train, X_test, y_train, y_test, and the fitted scaler.
    """
    # 1. Load data
    df = load_data(data_path)
    
    # 2. Clean data
    df = clean_data(df)
    
    # 3. Calibrate signal
    df = calibrate_signal(df)
    
    # 4. Define features
    features = ['concentration_calibrated']
    if include_env_features:
        features.extend(['temperature_C', 'pH', 'flow_rate_Ls'])
        
    # Sort chronologically to ensure no data leakage and correct time sequences
    df = df.sort_values(by=['timestamp', 'site', 'compound'])
    
    # 5. Split train/test chronologically
    unique_timestamps = sorted(df['timestamp'].unique())
    split_idx = int(len(unique_timestamps) * config.TRAIN_SPLIT)
    train_end_time = unique_timestamps[split_idx - 1]
    
    train_df = df[df['timestamp'] <= train_end_time].copy()
    test_df = df[df['timestamp'] > train_end_time].copy()
    
    # 6. Normalize
    scaler = MinMaxScaler()
    # Fit ONLY on train set to prevent data leakage
    scaler.fit(train_df[features])
    
    # Save scaler
    data_dir = os.path.dirname(os.path.abspath(data_path))
    scaler_path = os.path.join(data_dir, scaler_filename)
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
        
    # Transform both train and test sets
    train_df.loc[:, features] = scaler.transform(train_df[features])
    test_df.loc[:, features] = scaler.transform(test_df[features])
    
    # 7. Create sliding windows per site and compound
    target_idx = features.index('concentration_calibrated')
    window_size = config.WINDOW_SIZE
    forecast_hours = config.FORECAST_HOURS
    max_horizon = max(forecast_hours) if forecast_hours else 1
    
    def create_windows(df_group: pd.DataFrame) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        data = df_group[features].values
        X, y = [], []
        
        if forecast_hours:
            for i in range(len(data) - window_size - max_horizon + 1):
                X.append(data[i : i + window_size])
                y.append([data[i + window_size - 1 + h, target_idx] for h in forecast_hours])
        else:
            for i in range(len(data) - window_size):
                X.append(data[i : i + window_size])
                y.append([data[i + window_size, target_idx]])
                
        return X, y
        
    X_train_list, y_train_list = [], []
    for _, group in train_df.groupby(['site', 'compound']):
        group = group.sort_values('timestamp')
        X, y = create_windows(group)
        X_train_list.extend(X)
        y_train_list.extend(y)
        
    X_test_list, y_test_list = [], []
    for _, group in test_df.groupby(['site', 'compound']):
        group = group.sort_values('timestamp')
        X, y = create_windows(group)
        X_test_list.extend(X)
        y_test_list.extend(y)
        
    X_train = np.array(X_train_list) if X_train_list else np.array([])
    y_train = np.array(y_train_list) if y_train_list else np.array([])
    X_test = np.array(X_test_list) if X_test_list else np.array([])
    y_test = np.array(y_test_list) if y_test_list else np.array([])
    
    if len(X_train) < 500 and config.WINDOW_SIZE == 24:
        print(f"Warning: X_train has only {len(X_train)} samples. Reducing WINDOW_SIZE from 24 to 12 and re-processing...")
        config.WINDOW_SIZE = 12
        return prepare_data(data_path, include_env_features, scaler_filename)
    
    # 8. Print shapes
    print("--- Data Processing Summary ---")
    print(f"X_train shape: {X_train.shape}")
    print(f"y_train shape: {y_train.shape}")
    print(f"X_test shape: {X_test.shape}")
    print(f"y_test shape: {y_test.shape}")
    
    return X_train, X_test, y_train, y_test, scaler


if __name__ == "__main__":
    # Quick test when run directly
    import os
    test_data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), config.RUNS_DIR, "test_run", "pfas_sensor_data.csv")
    if os.path.exists(test_data_path):
        prepare_data(test_data_path, include_env_features=True)
    else:
        print(f"No test data found at {test_data_path}. Run data_simulation.py first.")
