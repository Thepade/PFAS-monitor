import os
import base64
from typing import Dict, Any
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import config

# Use the requested seaborn style
plt.style.use('seaborn-v0_8-whitegrid')


def plot_timeseries(df: pd.DataFrame, output_dir: str) -> None:
    """Multi-line PFAS over time for all sites with anomalies highlighted."""
    plt.figure(figsize=(14, 7))
    
    # Resolve actual column names defensively
    site_col = 'site_id' if 'site_id' in df.columns else (
        'site' if 'site' in df.columns else (
        'location' if 'location' in df.columns else df.columns[0]))

    compound_col = 'compound' if 'compound' in df.columns else (
        'contaminant' if 'contaminant' in df.columns else (
        'chemical' if 'chemical' in df.columns else None))

    conc_col = 'concentration_calibrated' if 'concentration_calibrated' in df.columns else (
        'concentration_ppt' if 'concentration_ppt' in df.columns else (
        'concentration' if 'concentration' in df.columns else None))

    if conc_col is None:
        raise ValueError(f"No concentration column found. Available: {list(df.columns)}")
    
    # Plot timeseries lines
    sns.lineplot(data=df, x='timestamp', y=conc_col, hue=site_col, style=compound_col, alpha=0.7)
    
    # Plot anomalies
    if 'anomaly_flag' in df.columns:
        anomalies = df[df['anomaly_flag'] == True]
        if not anomalies.empty:
            plt.scatter(anomalies['timestamp'], anomalies[conc_col], color='red', s=40, zorder=5, label='Anomaly')
            
    # EPA threshold lines
    plt.axhline(y=config.EPA_LIMIT_PPT, color='red', linestyle='--', linewidth=2, label=f'EPA Limit ({config.EPA_LIMIT_PPT} ppt)')
    plt.axhline(y=config.EPA_ADVISORY_PPT, color='orange', linestyle='--', linewidth=2, label=f'EPA Advisory ({config.EPA_ADVISORY_PPT} ppt)')
    
    plt.title('PFAS Concentration Over Time')
    plt.xlabel('Date')
    plt.ylabel('Concentration (ppt)')
    
    # Clean up legend (remove duplicate labels)
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(), loc='upper left', bbox_to_anchor=(1.02, 1))
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'pfas_concentration_timeseries.png'), dpi=config.FIGURE_DPI)
    plt.close()


def plot_predicted_vs_actual(predictions_dict: Dict[str, np.ndarray], metrics_dict: Dict[str, Dict[str, float]], output_dir: str) -> None:
    """Scatter of predicted vs actual for all models with 45-degree reference line."""
    if 'Actual' not in predictions_dict:
        print("Warning: 'Actual' key missing in predictions_dict. Skipping predicted vs actual plot.")
        return
        
    y_actual = predictions_dict['Actual']
    # If multi-horizon, take the first horizon (e.g., 24h) for the scatter
    y_actual_plot = y_actual[:, 0] if (y_actual.ndim > 1 and y_actual.shape[1] > 0) else y_actual
        
    plt.figure(figsize=(10, 8))
    
    colors = ['#3498db', '#2ecc71', '#9b59b6', '#e67e22']
    c_idx = 0
    min_val, max_val = float('inf'), float('-inf')
    
    for model_name, y_pred in predictions_dict.items():
        if model_name == 'Actual':
            continue
            
        y_pred_plot = y_pred[:, 0] if (y_pred.ndim > 1 and y_pred.shape[1] > 0) else y_pred
            
        metrics = metrics_dict.get(model_name, {})
        rmse = metrics.get('RMSE', 0.0)
        r2 = metrics.get('R2', 0.0)
        
        plt.scatter(y_actual_plot, y_pred_plot, alpha=0.5, label=f"{model_name} (RMSE={rmse:.2f}, R²={r2:.2f})", color=colors[c_idx % len(colors)])
        c_idx += 1
        
        min_val = min(min_val, y_actual_plot.min(), y_pred_plot.min())
        max_val = max(max_val, y_actual_plot.max(), y_pred_plot.max())
        
    # 45 degree reference line
    if min_val != float('inf'):
        plt.plot([min_val, max_val], [min_val, max_val], 'k--', label='Perfect Prediction')
        
    plt.title('Predicted vs Actual PFAS Concentration (1st Horizon)')
    plt.xlabel('Actual Concentration (ppt)')
    plt.ylabel('Predicted Concentration (ppt)')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'predicted_vs_actual.png'), dpi=config.FIGURE_DPI)
    plt.close()


def plot_forecast(df: pd.DataFrame, predictions_dict: Dict[str, np.ndarray], metrics_dict: Dict[str, Dict[str, float]], best_model_name: str, output_dir: str) -> None:
    """Plot last 7 days + 72hr forecast with ±1 std confidence band."""
    # Resolve actual column names defensively
    site_col = 'site_id' if 'site_id' in df.columns else (
        'site' if 'site' in df.columns else (
        'location' if 'location' in df.columns else df.columns[0]))

    compound_col = 'compound' if 'compound' in df.columns else (
        'contaminant' if 'contaminant' in df.columns else (
        'chemical' if 'chemical' in df.columns else None))

    conc_col = 'concentration_calibrated' if 'concentration_calibrated' in df.columns else (
        'concentration_ppt' if 'concentration_ppt' in df.columns else (
        'concentration' if 'concentration' in df.columns else None))

    if conc_col is None:
        raise ValueError(f"No concentration column found. Available: {list(df.columns)}")
    
    # System wide mean for historical context
    df_agg = df.groupby('timestamp')[conc_col].mean().reset_index()
    last_time = df_agg['timestamp'].max()
    history = df_agg[df_agg['timestamp'] > last_time - pd.Timedelta(days=config.PLOT_HISTORICAL_DAYS)]
    
    avg_forecast = None
    residual_std = None
    
    if best_model_name in predictions_dict:
        preds = predictions_dict[best_model_name]
        if preds.ndim > 1 and preds.shape[1] > 0:
            # Average the last num_series predictions
            num_series = len(config.SITES) * len(config.COMPOUNDS)
            last_preds = preds[-num_series:] if len(preds) >= num_series else preds
            avg_forecast = last_preds.mean(axis=0)
            
            if 'Actual' in predictions_dict:
                residuals = predictions_dict['Actual'] - predictions_dict[best_model_name]
                residual_std = np.std(residuals, axis=0)
            else:
                residual_std = np.zeros_like(avg_forecast)
                
    if avg_forecast is not None:
        forecast_times = [last_time + pd.Timedelta(hours=h) for h in config.FORECAST_HOURS]
        
        plt.figure(figsize=(12, 6))
        plt.plot(history['timestamp'], history[conc_col], label='Historical (Avg)', color='black')
        
        plt.plot(forecast_times, avg_forecast, label=f'Forecast ({best_model_name})', color='blue', marker='o')
        plt.fill_between(forecast_times, avg_forecast - residual_std, avg_forecast + residual_std, color='blue', alpha=0.2, label='±1 Std Dev Confidence Band')
        
        plt.axvline(x=last_time, color='gray', linestyle='--', label='Forecast start')
        
        plt.title(f'System-Wide PFAS Forecast ({config.PLOT_HISTORICAL_DAYS} Days History + 72hr Forecast)')
        plt.xlabel('Date')
        plt.ylabel('Concentration (ppt)')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, '72hr_forecast.png'), dpi=config.FIGURE_DPI)
        plt.close()


def plot_anomaly_heatmap(df: pd.DataFrame, output_dir: str) -> None:
    """Seaborn heatmap with sites on Y, weekly bins on X, cell = anomaly count."""
    # Resolve actual column names defensively
    site_col = 'site_id' if 'site_id' in df.columns else (
        'site' if 'site' in df.columns else (
        'location' if 'location' in df.columns else df.columns[0]))

    compound_col = 'compound' if 'compound' in df.columns else (
        'contaminant' if 'contaminant' in df.columns else (
        'chemical' if 'chemical' in df.columns else None))

    conc_col = 'concentration_calibrated' if 'concentration_calibrated' in df.columns else (
        'concentration_ppt' if 'concentration_ppt' in df.columns else (
        'concentration' if 'concentration' in df.columns else None))

    if conc_col is None:
        raise ValueError(f"No concentration column found. Available: {list(df.columns)}")
    if 'anomaly_flag' not in df.columns:
        return
        
    df_anom = df[df['anomaly_flag'] == True].copy()
    if df_anom.empty:
        plt.figure(figsize=(10, 6))
        plt.text(0.5, 0.5, 'No Anomalies Detected', ha='center', va='center')
        plt.savefig(os.path.join(output_dir, 'anomaly_heatmap.png'), dpi=config.FIGURE_DPI)
        plt.close()
        return
        
    # Weekly bins
    df_anom['week'] = df_anom['timestamp'].dt.to_period('W').apply(lambda r: r.start_time)
    
    pivot = pd.crosstab(df_anom[site_col], df_anom['week'])
    pivot.columns = [ts.strftime('%Y-%m-%d') for ts in pivot.columns]
    
    plt.figure(figsize=(12, 6))
    sns.heatmap(pivot, cmap='YlOrRd', annot=True, fmt='d', linewidths=.5)
    plt.title('Weekly Anomalies by Site')
    plt.xlabel('Week Starting')
    plt.ylabel('Site')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'anomaly_heatmap.png'), dpi=config.FIGURE_DPI)
    plt.close()


def plot_residuals(predictions_dict: Dict[str, np.ndarray], output_dir: str) -> None:
    """One subplot per model, histogram with KDE of (Actual – Predicted)."""
    if 'Actual' not in predictions_dict:
        return
        
    y_actual = predictions_dict['Actual']
    models = [m for m in predictions_dict.keys() if m != 'Actual']
    
    if not models:
        return
        
    fig, axes = plt.subplots(len(models), 1, figsize=(10, 4 * len(models)))
    if len(models) == 1:
        axes = [axes]
        
    for ax, model_name in zip(axes, models):
        y_pred = predictions_dict[model_name]
        
        # Flatten multi-horizon for overall residuals distribution
        res = (y_actual - y_pred).flatten()
        
        sns.histplot(res, kde=True, ax=ax, color='#3498db')
        ax.axvline(x=0, color='red', linestyle='--', linewidth=2)
        
        mean_res, std_res = np.mean(res), np.std(res)
        ax.text(0.05, 0.95, f"Mean: {mean_res:.2f}\nStd: {std_res:.2f}", 
                transform=ax.transAxes, verticalalignment='top', 
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
                
        ax.set_title(f'Residuals Distribution: {model_name}')
        ax.set_xlabel('Residual (Actual - Predicted) (ppt)')
        ax.set_ylabel('Frequency')
        
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'residual_plot.png'), dpi=config.FIGURE_DPI)
    plt.close()


def generate_all_plots(df: pd.DataFrame, predictions_dict: Dict[str, np.ndarray], metrics_dict: Dict[str, Dict[str, float]], best_model_name: str, output_dir: str) -> None:
    """Calls all 5 plot functions in order."""
    os.makedirs(output_dir, exist_ok=True)
    plot_timeseries(df, output_dir)
    plot_predicted_vs_actual(predictions_dict, metrics_dict, output_dir)
    plot_forecast(df, predictions_dict, metrics_dict, best_model_name, output_dir)
    plot_anomaly_heatmap(df, output_dir)
    plot_residuals(predictions_dict, output_dir)


def save_metrics_csv(metrics_dict: Dict[str, Dict[str, float]], output_dir: str) -> None:
    """Saves DataFrame with columns Model, RMSE, MAE, R2 rounded to 4 dp."""
    os.makedirs(output_dir, exist_ok=True)
    records = []
    for model, metrics in metrics_dict.items():
        records.append({
            'Model': model,
            'RMSE': round(metrics.get('RMSE', 0.0), 4),
            'MAE': round(metrics.get('MAE', 0.0), 4),
            'R2': round(metrics.get('R2', 0.0), 4)
        })
    df_metrics = pd.DataFrame(records)
    df_metrics.to_csv(os.path.join(output_dir, 'metrics_summary.csv'), index=False)
