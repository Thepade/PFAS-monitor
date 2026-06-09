import os
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest

import config


def detect_anomalies(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply Isolation Forest to detect anomalies and assign severity 
    based on EPA threshold rules.
    
    Args:
        df (pd.DataFrame): The input dataframe containing PFAS readings.
        
    Returns:
        pd.DataFrame: DataFrame with anomaly_flag and severity columns added.
    """
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
        
    df_out = df.copy()
    
    # 1. Isolation Forest for statistical anomaly detection
    # Use numeric columns for the isolation forest to detect multivariable anomalies
    numeric_cols = df_out.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        iso_forest = IsolationForest(
            contamination=config.ANOMALY_CONTAMINATION,
            random_state=config.RANDOM_SEED
        )
        # fit_predict returns 1 for inliers, -1 for outliers
        preds = iso_forest.fit_predict(df_out[numeric_cols].fillna(0))
        df_out['anomaly_flag'] = (preds == -1)
    else:
        df_out['anomaly_flag'] = False
        
    # 2. EPA threshold rules for severity
    def assign_severity(val: float) -> str:
        if pd.isna(val):
            return "normal"
        if val >= config.EPA_LIMIT_PPT:
            return "critical"
        elif val >= config.EPA_ADVISORY_PPT:
            return "elevated"
        else:
            return "normal"
            
    df_out['severity'] = df_out[conc_col].apply(assign_severity)
    
    # Print summary statistics
    total_anomalies = int(df_out['anomaly_flag'].sum())
    
    print("\n--- Anomaly Detection Summary ---")
    print(f"Total Anomalies (Isolation Forest) : {total_anomalies}")
    
    if site_col in df_out.columns:
        print("\n[Anomalies by Site]")
        site_counts = df_out[df_out['anomaly_flag']][site_col].value_counts()
        if site_counts.empty:
            print("  - None")
        else:
            for site, count in site_counts.items():
                print(f"  - {site}: {count}")
            
    print("\n[Count by Severity Level]")
    severity_counts = df_out['severity'].value_counts()
    for sev, count in severity_counts.items():
        print(f"  - {sev}: {count}")
        
    return df_out


if __name__ == "__main__":
    # Test script locally
    test_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), config.RUNS_DIR, "test_run", "pfas_sensor_data.csv")
    if os.path.exists(test_path):
        from data_processing import load_data, clean_data, calibrate_signal
        
        print("Loading test data for anomaly detection...")
        df_test = load_data(test_path)
        df_test = clean_data(df_test)
        df_test = calibrate_signal(df_test)
        
        result_df = detect_anomalies(df_test)
    else:
        print("No test data found. Run data_simulation.py first.")
