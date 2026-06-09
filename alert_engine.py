import numpy as np
import pandas as pd
from typing import Dict

import config
import security


def calculate_trend(series: pd.Series) -> str:
    """Calculate the trend (Rising/Stable/Falling) based on recent readings."""
    if len(series) < 2:
        return "Stable"
    
    y = series.values
    x = np.arange(len(y))
    slope, _ = np.polyfit(x, y, 1)
    
    if slope > config.TREND_THRESHOLD:
        return "Rising"
    elif slope < -config.TREND_THRESHOLD:
        return "Falling"
    else:
        return "Stable"


def determine_risk(val: float) -> str:
    """Determine risk level based on EPA thresholds."""
    if val >= config.EPA_LIMIT_PPT:
        return "High"
    elif val >= config.EPA_ADVISORY_PPT:
        return "Medium"
    else:
        return "Low"


def format_line(text: str, width: int = 48) -> str:
    """Helper to format a line inside the box."""
    return f"│ {text.ljust(width - 2)} │"


def generate_alerts(df: pd.DataFrame, site_forecasts: Dict[str, Dict[str, float]], output_dir: str) -> None:
    """
    Generate alert reports and simulate signed alerts for critical occurrences.
    """
    print("\n" + "="*50)
    print(" PFAS RISK & ALERT REPORT ".center(50, "="))
    print("="*50 + "\n")
    
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
    
    for site in config.SITES:
        for compound in config.COMPOUNDS:
            site_df = df[(df[site_col] == site) & (df[compound_col] == compound)].copy()
            if site_df.empty:
                continue
                
            site_df = site_df.sort_values('timestamp')
            
            current_val = site_df[conc_col].iloc[-1]
            risk_level = determine_risk(current_val)
            
            recent_vals = site_df[conc_col].tail(config.TREND_WINDOW)
            trend = calculate_trend(recent_vals)
            
            forecast_val = site_forecasts.get(site, {}).get(compound, 0.0)
            
            # Print formatted box-drawing risk report
            header = f"─ SITE: {site} | COMPOUND: {compound} "
            print(f"┌{header.ljust(48, '─')}┐")
            print(format_line(f"Current Conc : {current_val:.2f} ppt"))
            print(format_line(f"24h Forecast : {forecast_val:.2f} ppt"))
            print(format_line(f"Risk Level   : {risk_level}"))
            print(format_line(f"Trend        : {trend}"))
            print(f"└{'─'*48}┘")
            
            # Critical severity checks
            if risk_level == "High":
                msg = f"CRITICAL: {compound} at {site} exceeds EPA limit ({current_val:.2f} ppt)."
                signed_alert = security.sign_alert(msg, site, compound)
                is_valid = security.verify_signature(signed_alert)
                
                print(f"  [SECURITY] Alert signature verified: {is_valid}")
                print(f"  [SMS ALERT] ALARM! {msg} Immediate action required.\n")
                
                security.log_security_event("CRITICAL_ALERT", msg, output_dir)
