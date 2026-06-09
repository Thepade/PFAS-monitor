import os
import random
from typing import List, Optional, Tuple, Dict
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

import config

# Set module-level random seeds for reproducibility
np.random.seed(config.RANDOM_SEED)
random.seed(config.RANDOM_SEED)


def simulate(
    days: Optional[int] = None,
    sites: Optional[List[str]] = None,
    noise_std: Optional[float] = None
) -> Tuple[pd.DataFrame, Dict[str, Dict[str, int]]]:
    """
    Simulate hourly PFAS readings with baseline, seasonal variation, noise,
    and industrial discharge spikes.
    
    Args:
        days (int, optional): Number of days to simulate. Defaults to config.SIMULATION_DAYS.
        sites (List[str], optional): List of site names. Defaults to config.SITES.
        noise_std (float, optional): Standard deviation of noise. Defaults to config.NOISE_STD.
        
    Returns:
        Tuple[pd.DataFrame, Dict[str, Dict[str, int]]]: 
            - DataFrame containing the simulated data.
            - Dictionary containing spike counts per site and compound.
    """
    days_to_sim = days if days is not None else config.SIMULATION_DAYS
    sites_to_sim = sites if sites is not None else config.SITES
    noise_to_sim = noise_std if noise_std is not None else config.NOISE_STD
    
    # Calculate timestamps (ending now, starting `days` ago)
    end_date = datetime.now().replace(minute=0, second=0, microsecond=0)
    start_date = end_date - timedelta(days=days_to_sim)
    
    # Generate hourly timestamps based on sampling interval
    freq_str = f"{config.SAMPLING_INTERVAL_HOURS}h"
    timestamps = pd.date_range(start=start_date, end=end_date, freq=freq_str)
    
    n_points = len(timestamps)
    records = []
    spike_summary: Dict[str, Dict[str, int]] = {}
    
    for site in sites_to_sim:
        spike_summary[site] = {}
        for compound in config.COMPOUNDS:
            # Baseline concentration
            baseline = np.random.uniform(config.BASELINE_MIN, config.BASELINE_MAX)
            
            # Seasonal variation (combination of daily and weekly sinusoids)
            time_idx = np.arange(n_points)
            # 24h cycle
            daily_seasonality = 2.0 * np.sin(2 * np.pi * time_idx / (24 / config.SAMPLING_INTERVAL_HOURS))
            # 7-day cycle
            weekly_seasonality = 1.0 * np.sin(2 * np.pi * time_idx / (24 * 7 / config.SAMPLING_INTERVAL_HOURS))
            
            # Gaussian noise for concentration
            noise = np.random.normal(0, noise_to_sim, n_points)
            
            # Base concentration profile
            concentration = baseline + daily_seasonality + weekly_seasonality + noise
            concentration = np.clip(concentration, 0.0, None)
            
            # Inject industrial discharge spikes
            num_spikes = random.randint(2, 3)
            spike_summary[site][compound] = num_spikes
            
            for _ in range(num_spikes):
                spike_dur = random.randint(config.SPIKE_DURATION_HOURS[0], config.SPIKE_DURATION_HOURS[1])
                # Ensure spike doesn't exceed array bounds
                if n_points - spike_dur - 1 > 0:
                    spike_start = random.randint(0, n_points - spike_dur - 1)
                    spike_end = spike_start + spike_dur
                    
                    spike_val = np.random.uniform(config.SPIKE_CONCENTRATION[0], config.SPIKE_CONCENTRATION[1])
                    concentration[spike_start:spike_end] += spike_val
            
            # Environmental covariates
            temp_c = np.random.uniform(15.0, 35.0, n_points)
            ph = np.random.uniform(6.5, 8.5, n_points)
            flow_rate_ls = np.random.uniform(0.5, 5.0, n_points)
            
            # Raw biosensor signal column: signal_mV = (concentration - B) / A + noise
            signal_noise = np.random.normal(0, noise_to_sim, n_points)
            signal_mv = (concentration - config.CALIBRATION_B) / config.CALIBRATION_A + signal_noise
            
            # Create DataFrame block for this site/compound
            df_block = pd.DataFrame({
                "timestamp": timestamps,
                "site": site,
                "compound": compound,
                "temperature_C": temp_c,
                "pH": ph,
                "flow_rate_Ls": flow_rate_ls,
                "concentration_ppt": concentration,
                "signal_mV": signal_mv
            })
            records.append(df_block)
            
    # Combine all blocks
    df_final = pd.concat(records, ignore_index=True)
    
    # Sort chronologically, then by site and compound for neatness
    df_final = df_final.sort_values(by=["timestamp", "site", "compound"]).reset_index(drop=True)
    
    return df_final, spike_summary


def main(
    output_dir: str,
    days: Optional[int] = None,
    sites: Optional[List[str]] = None,
    noise_std: Optional[float] = None
) -> str:
    """
    Main orchestrator function for the data simulation module.
    
    Args:
        output_dir (str): Directory where the output CSV will be saved.
        days (int, optional): Number of days to simulate.
        sites (List[str], optional): Sites to simulate.
        noise_std (float, optional): Standard deviation of noise.
        
    Returns:
        str: Absolute path to the generated CSV file.
    """
    print("--- Starting PFAS Data Simulation ---")
    
    # Run simulation
    df, spike_counts = simulate(days=days, sites=sites, noise_std=noise_std)
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Save to CSV
    output_filepath = os.path.join(output_dir, "pfas_sensor_data.csv")
    df.to_csv(output_filepath, index=False)
    
    # Print summary statistics
    total_rows = len(df)
    columns = list(df.columns)
    unique_sites = df["site"].unique().tolist()
    unique_compounds = df["compound"].unique().tolist()
    min_date = df["timestamp"].min()
    max_date = df["timestamp"].max()
    
    print("\n[Simulation Summary]")
    print(f"File Saved      : {output_filepath}")
    print(f"Total Rows      : {total_rows}")
    print(f"Columns         : {columns}")
    print(f"Sites           : {unique_sites}")
    print(f"Compounds       : {unique_compounds}")
    print(f"Date Range      : {min_date} to {max_date}")
    
    print("\n[Spikes Injected per Site]")
    for site, compound_spikes in spike_counts.items():
        print(f"  {site}:")
        for compound, counts in compound_spikes.items():
            print(f"    - {compound}: {counts} spikes")
            
    print("\n--- Data Simulation Complete ---")
    return output_filepath


if __name__ == "__main__":
    # Quick test when run directly
    test_out = os.path.join(os.path.dirname(os.path.abspath(__file__)), config.RUNS_DIR, "test_run")
    main(output_dir=test_out)
