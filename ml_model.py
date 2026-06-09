import os
import pickle
import random
import sys
from typing import Dict, Tuple, Any

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

import config
from data_processing import prepare_data

# Set random seeds for reproducibility
np.random.seed(config.RANDOM_SEED)
random.seed(config.RANDOM_SEED)

try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential, load_model
    from tensorflow.keras.layers import LSTM, Dense
    TF_AVAILABLE = True
    tf.random.set_seed(config.RANDOM_SEED)
except ImportError:
    TF_AVAILABLE = False


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Calculate regression metrics.
    
    Args:
        y_true (np.ndarray): True targets.
        y_pred (np.ndarray): Predicted targets.
        
    Returns:
        Dict[str, float]: Dictionary with RMSE, MAE, and R2.
    """
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    return {"RMSE": rmse, "MAE": mae, "R2": r2}


def inverse_transform_targets(y_scaled: np.ndarray, scaler: Any) -> np.ndarray:
    """
    Inverse transform just the target column.
    
    Args:
        y_scaled (np.ndarray): Scaled predictions or targets, shape (N, horizons).
        scaler (Any): Fitted MinMaxScaler from prepare_data.
        
    Returns:
        np.ndarray: Inverse transformed array.
    """
    n_features = scaler.n_features_in_
    # Target is index 0 based on prepare_data in data_processing.py
    target_idx = 0
    
    if y_scaled.ndim == 1:
        y_scaled = y_scaled.reshape(-1, 1)
        
    y_inv = np.zeros_like(y_scaled)
    for i in range(y_scaled.shape[1]):
        dummy = np.zeros((len(y_scaled), n_features))
        dummy[:, target_idx] = y_scaled[:, i]
        inv = scaler.inverse_transform(dummy)
        y_inv[:, i] = inv[:, target_idx]
    return y_inv


def train_and_eval_sklearn(
    model: Any, 
    X_train: np.ndarray, y_train: np.ndarray, 
    X_test: np.ndarray, y_test: np.ndarray, 
    scaler: Any, 
    model_path: str,
    model_name: str
) -> Tuple[Any, np.ndarray, Dict[str, float]]:
    """
    Train and evaluate a scikit-learn model, utilizing caching if available.
    """
    if os.path.exists(model_path):
        print(f"[CACHE] Loaded existing {model_name} model — skipping retraining")
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
    else:
        model.fit(X_train, y_train)
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
            
    # Generate iterative/multi-horizon forecasts
    y_pred = model.predict(X_test)
    
    y_test_inv = inverse_transform_targets(y_test, scaler)
    y_pred_inv = inverse_transform_targets(y_pred, scaler)
    
    metrics = calculate_metrics(y_test_inv, y_pred_inv)
    return model, y_pred_inv, metrics


def train_and_eval_lstm(
    X_train: np.ndarray, y_train: np.ndarray, 
    X_test: np.ndarray, y_test: np.ndarray, 
    scaler: Any, 
    model_path: str
) -> Tuple[Any, np.ndarray, Dict[str, float]]:
    """
    Train and evaluate LSTM model, utilizing caching if available.
    """
    if os.path.exists(model_path):
        print(f"[CACHE] Loaded existing LSTM model — skipping retraining")
        model = load_model(model_path)
    else:
        model = Sequential([
            LSTM(config.LSTM_UNITS, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
            LSTM(config.LSTM_UNITS),
            Dense(y_train.shape[1])
        ])
        model.compile(optimizer='adam', loss='mse')
        model.fit(
            X_train, y_train, 
            epochs=config.LSTM_EPOCHS, 
            batch_size=config.LSTM_BATCH_SIZE, 
            validation_split=config.LSTM_VAL_SPLIT,
            verbose=0
        )
        model.save(model_path)
        
    # Generate iterative/multi-horizon forecasts
    y_pred = model.predict(X_test, verbose=0)
    
    y_test_inv = inverse_transform_targets(y_test, scaler)
    y_pred_inv = inverse_transform_targets(y_pred, scaler)
    
    metrics = calculate_metrics(y_test_inv, y_pred_inv)
    return model, y_pred_inv, metrics


def train_models(data_path: str, output_dir: str) -> Tuple[Dict[str, np.ndarray], Dict[str, Dict[str, float]], str, Dict[str, Dict[str, float]]]:
    """
    Train and evaluate all models, perform ablation study, and return results.
    
    Args:
        data_path (str): Path to the processed CSV data.
        output_dir (str): Directory to save model artifacts.
        
    Returns:
        Tuple: predictions_dict, metrics_dict, best_model_name, ablation_results.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Prepare Primary Data
    print("--- Preparing Data (with env features) ---")
    X_train, X_test, y_train, y_test, scaler = prepare_data(data_path, include_env_features=True)
    
    # Reshape for sklearn models (flatten windows)
    X_train_flat = X_train.reshape(X_train.shape[0], -1)
    X_test_flat = X_test.reshape(X_test.shape[0], -1)
    
    predictions_dict = {}
    metrics_dict = {}
    
    # Model 1: Linear Regression
    lr_path = os.path.join(output_dir, "lr_model.pkl")
    lr = LinearRegression()
    _, y_pred_lr, metrics_lr = train_and_eval_sklearn(
        lr, X_train_flat, y_train, X_test_flat, y_test, scaler, lr_path, "LR"
    )
    predictions_dict["Linear Regression"] = y_pred_lr
    metrics_dict["Linear Regression"] = metrics_lr
    
    # Model 2: Random Forest
    rf_path = os.path.join(output_dir, "rf_model.pkl")
    rf = RandomForestRegressor(
        n_estimators=config.RF_ESTIMATORS, 
        max_depth=config.RF_MAX_DEPTH, 
        random_state=config.RANDOM_SEED
    )
    _, y_pred_rf, metrics_rf = train_and_eval_sklearn(
        rf, X_train_flat, y_train, X_test_flat, y_test, scaler, rf_path, "RF"
    )
    predictions_dict["Random Forest"] = y_pred_rf
    metrics_dict["Random Forest"] = metrics_rf
    
    # Model 3: LSTM (Optional)
    if TF_AVAILABLE:
        lstm_path = os.path.join(output_dir, "lstm_model.h5")
        _, y_pred_lstm, metrics_lstm = train_and_eval_lstm(
            X_train, y_train, X_test, y_test, scaler, lstm_path
        )
        predictions_dict["LSTM"] = y_pred_lstm
        metrics_dict["LSTM"] = metrics_lstm
        
    # 2. Ablation Study
    print("\n--- Running Ablation Study (without env features) ---")
    X_train_abl, X_test_abl, y_train_abl, y_test_abl, scaler_abl = prepare_data(data_path, include_env_features=False, scaler_filename="scaler_ablation.pkl")
    
    X_train_abl_flat = X_train_abl.reshape(X_train_abl.shape[0], -1)
    X_test_abl_flat = X_test_abl.reshape(X_test_abl.shape[0], -1)
    
    rf_abl_path = os.path.join(output_dir, "rf_model_ablation.pkl")
    rf_abl = RandomForestRegressor(
        n_estimators=config.RF_ESTIMATORS, 
        max_depth=config.RF_MAX_DEPTH, 
        random_state=config.RANDOM_SEED
    )
    _, _, metrics_rf_abl = train_and_eval_sklearn(
        rf_abl, X_train_abl_flat, y_train_abl, X_test_abl_flat, y_test_abl, scaler_abl, rf_abl_path, "RF (Ablation)"
    )
    
    ablation_results = {
        "with_env": metrics_rf,
        "without_env": metrics_rf_abl
    }
    
    # Print Ablation Table
    print("\n┌─────────────────────────────┬────────┬────────┬────────┐")
    print("│ Configuration               │  RMSE  │  MAE   │   R²   │")
    print("├─────────────────────────────┼────────┼────────┼────────┤")
    print(f"│ RF with env. features       │ {metrics_rf['RMSE']:6.2f} │ {metrics_rf['MAE']:6.2f} │ {metrics_rf['R2']:6.2f} │")
    print(f"│ RF without env. features    │ {metrics_rf_abl['RMSE']:6.2f} │ {metrics_rf_abl['MAE']:6.2f} │ {metrics_rf_abl['R2']:6.2f} │")
    print("└─────────────────────────────┴────────┴────────┴────────┘")
    
    rmse_diff = metrics_rf_abl['RMSE'] - metrics_rf['RMSE']
    rmse_pct = (rmse_diff / metrics_rf_abl['RMSE']) * 100 if metrics_rf_abl['RMSE'] != 0 else 0.0
    print(f"\nEnvironmental features improved RMSE by {rmse_diff:.2f} ppt ({rmse_pct:.1f}%)")

    # Print Full Model Comparison Table
    print("\n┌────────────────────┬────────┬────────┬────────┐")
    print("│ Model              │  RMSE  │  MAE   │   R²   │")
    print("├────────────────────┼────────┼────────┼────────┤")
    print(f"│ Linear Regression  │ {metrics_lr['RMSE']:6.2f} │ {metrics_lr['MAE']:6.2f} │ {metrics_lr['R2']:6.2f} │")
    print(f"│ Random Forest      │ {metrics_rf['RMSE']:6.2f} │ {metrics_rf['MAE']:6.2f} │ {metrics_rf['R2']:6.2f} │")
    if TF_AVAILABLE:
        print(f"│ LSTM               │ {metrics_lstm['RMSE']:6.2f} │ {metrics_lstm['MAE']:6.2f} │ {metrics_lstm['R2']:6.2f} │")
    else:
        print("│ LSTM               │   N/A  │   N/A  │   N/A  │")
    print("└────────────────────┴────────┴────────┴────────┘")
    
    # Determine best model
    best_model_name = min(metrics_dict, key=lambda k: metrics_dict[k]["RMSE"])
    
    return predictions_dict, metrics_dict, best_model_name, ablation_results


if __name__ == "__main__":
    # Test script locally
    test_data = os.path.join(os.path.dirname(os.path.abspath(__file__)), config.RUNS_DIR, "test_run", "pfas_sensor_data.csv")
    test_out = os.path.join(os.path.dirname(os.path.abspath(__file__)), config.RUNS_DIR, "test_run")
    if os.path.exists(test_data):
        train_models(test_data, test_out)
    else:
        print("No test data found. Run data_simulation.py first.")
