import os
import base64
from datetime import datetime
from typing import Dict, Any
import numpy as np

import config


def generate_text_report(
    metrics_dict: Dict[str, Dict[str, float]], 
    ablation_results: Dict[str, Dict[str, float]], 
    anomaly_summary: Dict[str, Any], 
    run_id: str, 
    output_dir: str,
    benchmark_results: list = None,
    data_source: str = "Simulation",
    env_results: Dict = None
) -> None:
    """Writes pfas_report.txt containing text summaries of the modeling run."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    report_path = os.path.join(output_dir, config.REPORT_FILENAME)
    
    lines = []
    lines.append("="*50)
    lines.append(f" PFAS MONITORING SYSTEM REPORT - {run_id}")
    lines.append(f" Data Source: {data_source}")
    lines.append(f" Generated: {timestamp}")
    lines.append("="*50 + "\n")
    
    # 1. Model Comparison
    lines.append("--- MODEL COMPARISON ---")
    for model, mets in metrics_dict.items():
        lines.append(f"{model}: RMSE={mets.get('RMSE',0):.2f}, MAE={mets.get('MAE',0):.2f}, R2={mets.get('R2',0):.2f}")
        
    # 2. Ablation Results
    lines.append("\n--- ABLATION STUDY ---")
    with_env = ablation_results.get('with_env', {})
    wo_env = ablation_results.get('without_env', {})
    lines.append(f"RF with Env Features    : RMSE={with_env.get('RMSE',0):.2f}")
    lines.append(f"RF without Env Features : RMSE={wo_env.get('RMSE',0):.2f}")
    diff = wo_env.get('RMSE', 0) - with_env.get('RMSE', 0)
    lines.append(f"Improvement             : {diff:.2f} ppt")
    
    # 3. Anomaly Summary
    lines.append("\n--- ANOMALY SUMMARY ---")
    lines.append(f"Total Anomalies: {anomaly_summary.get('total', 0)}")
    lines.append("By Site:")
    for s, c in anomaly_summary.get('by_site', {}).items():
        lines.append(f"  {s}: {c}")
    lines.append("By Severity:")
    for s, c in anomaly_summary.get('by_severity', {}).items():
        lines.append(f"  {s}: {c}")
        
    # 4. EPA Thresholds
    lines.append("\n--- EPA THRESHOLDS ---")
    lines.append(f"Limit    : {config.EPA_LIMIT_PPT} ppt")
    lines.append(f"Advisory : {config.EPA_ADVISORY_PPT} ppt")
    
    # 5. Generated Plots
    lines.append("\n--- GENERATED PLOTS ---")
    plots = [
        'pfas_concentration_timeseries.png',
        'predicted_vs_actual.png',
        '72hr_forecast.png',
        'anomaly_heatmap.png',
        'residual_plot.png'
    ]
    for p in plots:
        lines.append(f"- {p}")
        
    if benchmark_results:
        lines.append("\n=== SYSTEM PERFORMANCE BENCHMARKS ===")
        lines.append(f"{'Pipeline Stage':<27} | {'Time (sec)':<12} | {'Peak Memory (MB)':<16}")
        lines.append("-" * 65)
        for r in benchmark_results:
            lines.append(f"{r['stage']:<27} | {r['time_seconds']:<12.2f} | {r['peak_memory_mb']:<16.1f}")
            
    if env_results:
        lines.append("\n=== ENVIRONMENTAL RISK ASSESSMENT ===")
        lines.append("\n[EPA Exceedance Table]")
        for k, v in env_results.get('exceedance', {}).items():
            lines.append(f"{v['site']} {v['compound']}: N={v['n']}, >4ppt={v['gt_4']}({v['pct_4']:.1f}%), >70ppt={v['gt_70']}({v['pct_70']:.1f}%), Max={v['max']:.1f}")
        
        lines.append("\n[Seasonal Trend Summary per site]")
        for k, v in env_results.get('trend', {}).items():
            lines.append(f"{k}: Trend={v['trend_dir']}, Slope={v['trend_slope']:.2f}, Amp={v['seasonal_amp']:.1f}")
            
        lines.append("\n[Correlation Analysis Table]")
        for k, v in env_results.get('correlation', {}).items():
            lines.append(f"{k}: {v}")
            
        lines.append("\n[Model Statistical Significance Table]")
        for k, v in env_results.get('stats', {}).items():
            lines.append(f"{k}: Stat={v['stat']}, p={v['p']:.4f}")
            
        lines.append("\n[Full Environmental Interpretation narrative]")
        lines.append(env_results.get('narrative', ''))
            
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("\n".join(lines))


def generate_html_report(
    metrics_dict: Dict[str, Dict[str, float]], 
    ablation_results: Dict[str, Dict[str, float]], 
    anomaly_summary: Dict[str, Any], 
    predictions_dict: Dict[str, np.ndarray], 
    run_id: str, 
    output_dir: str,
    benchmark_results: list = None,
    data_source: str = "Simulation",
    env_results: Dict = None
) -> None:
    """Writes pfas_report.html containing a styled self-contained HTML report."""
    
    def get_base64_img(filename: str) -> str:
        filepath = os.path.join(output_dir, filename)
        if not os.path.exists(filepath):
            return ""
        with open(filepath, "rb") as img_file:
            b64_str = base64.b64encode(img_file.read()).decode('utf-8')
        return f"data:image/png;base64,{b64_str}"
        
    best_model = min(metrics_dict.keys(), key=lambda k: metrics_dict[k].get("RMSE", float('inf')))
    
    # Metrics rows
    metrics_html = ""
    for model, mets in metrics_dict.items():
        color = "#e8f5e9" if model == best_model else "#ffffff"
        metrics_html += f"<tr style='background-color: {color};'><td>{model}</td><td>{mets.get('RMSE',0):.2f}</td><td>{mets.get('MAE',0):.2f}</td><td>{mets.get('R2',0):.2f}</td></tr>"

    banner_html = ""
    if "Real Dataset" in data_source:
        banner_html = '''
        <div style="background:#d4edda; border:1px solid #28a745; padding:12px; border-radius:6px; margin:12px 0;">
          ✅ <strong>Validated on Real Field Data</strong> — Results are based on actual PFAS measurements, not simulation.
        </div>
        '''

    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <title>PFAS Monitor Report - {run_id}</title>
        <style>
            body {{ font-family: Arial, sans-serif; background-color: #f4f7f6; color: #333; margin: 0; padding: 20px; }}
            .container {{ max-width: 1200px; margin: 0 auto; background: white; padding: 30px; border-radius: 8px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }}
            h1, h2, h3 {{ color: #2c3e50; }}
            .header {{ text-align: center; border-bottom: 2px solid #3498db; padding-bottom: 20px; margin-bottom: 30px; }}
            table {{ width: 100%; border-collapse: collapse; margin-bottom: 30px; }}
            th, td {{ border: 1px solid #ddd; padding: 12px; text-align: left; }}
            th {{ background-color: #3498db; color: white; }}
            .card-container {{ display: flex; gap: 20px; margin-bottom: 30px; flex-wrap: wrap; }}
            .card {{ background: #ecf0f1; padding: 20px; border-radius: 8px; flex: 1; min-width: 200px; text-align: center; }}
            .card h3 {{ margin-top: 0; }}
            .img-container {{ margin-bottom: 40px; text-align: center; }}
            .img-container img {{ max-width: 100%; height: auto; border: 1px solid #ddd; border-radius: 4px; }}
            .footer {{ text-align: center; font-size: 12px; color: #7f8c8d; margin-top: 50px; border-top: 1px solid #eee; padding-top: 20px; }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>PFAS Monitoring System Report</h1>
                <p>Run ID: {run_id} | Data Source: {data_source} | Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
            </div>
            {banner_html}
            
            <h2>Model Evaluation</h2>
            <table>
                <tr><th>Model</th><th>RMSE (ppt)</th><th>MAE (ppt)</th><th>R² Score</th></tr>
                {metrics_html}
            </table>
            
            <h2>Anomaly Summary</h2>
            <div class="card-container">
                <div class="card">
                    <h3>Total Anomalies</h3>
                    <p style="font-size: 24px; font-weight: bold; color: #e74c3c;">{anomaly_summary.get('total', 0)}</p>
                </div>
    """
    
    for site, count in anomaly_summary.get('by_site', {}).items():
        html_content += f"""
                <div class="card">
                    <h3>{site}</h3>
                    <p style="font-size: 24px; font-weight: bold;">{count}</p>
                </div>
        """
        
    html_content += """
            </div>
    """
    
    # Forecast Values Table
    forecast_html = ""
    if best_model in predictions_dict:
        best_preds = predictions_dict[best_model]
        if best_preds.ndim > 1 and len(best_preds) > 0:
            last_forecast = best_preds[-1]
            if len(last_forecast) == len(config.FORECAST_HOURS):
                forecast_html = "<h2>System-Wide Forecast Values</h2><table><tr><th>Horizon</th><th>Predicted Concentration (ppt)</th></tr>"
                for h, val in zip(config.FORECAST_HOURS, last_forecast):
                    forecast_html += f"<tr><td>+{h} Hours</td><td>{val:.2f}</td></tr>"
                forecast_html += "</table>"
                
    html_content += forecast_html
    
    if benchmark_results:
        bench_html = "<h2>System Performance Benchmarks</h2><table><tr><th>Pipeline Stage</th><th>Time (sec)</th><th>Peak Memory (MB)</th></tr>"
        for r in benchmark_results:
            bench_html += f"<tr><td>{r['stage']}</td><td>{r['time_seconds']:.2f}</td><td>{r['peak_memory_mb']:.1f}</td></tr>"
        bench_html += "</table>"
        html_content += bench_html
        
    env_html = ""
    if env_results:
        env_html += "<h2>Environmental Risk Assessment</h2>"
        
        # Exceedance Table
        env_html += "<h3>EPA Regulatory Exceedance Summary</h3>"
        env_html += "<table><tr><th>Site</th><th>Compound</th><th>N</th><th>>4 ppt (Advisory)</th><th>>70 ppt (MCL)</th><th>Max (ppt)</th></tr>"
        for k, v in env_results.get('exceedance', {}).items():
            bg_color = "#f8d7da" if v['gt_70'] > 0 else ("#fff3cd" if v['gt_4'] > 0 else "#d4edda")
            env_html += f"<tr style='background-color: {bg_color};'><td>{v['site']}</td><td>{v['compound']}</td><td>{v['n']}</td><td>{v['gt_4']} ({v['pct_4']:.1f}%)</td><td>{v['gt_70']} ({v['pct_70']:.1f}%)</td><td>{v['max']:.1f}</td></tr>"
        env_html += "</table>"
        
        # Trend Badges
        env_html += "<h3>Seasonal Trend Summary</h3><div class='card-container'>"
        for k, v in env_results.get('trend', {}).items():
            badge = "🔴 INCREASING" if v['trend_dir'] == "INCREASING" else ("🟢 DECREASING" if v['trend_dir'] == "DECREASING" else "🟡 STABLE")
            env_html += f"<div class='card'><h3>{k}</h3><p style='font-size: 20px; font-weight: bold;'>{badge}</p><p>Slope: {v['trend_slope']:.2f} ppt/mo</p></div>"
        env_html += "</div>"
        
        # Compliance Cards
        env_html += "<h3>Regulatory Compliance Status (Last 30 Days)</h3><div class='card-container'>"
        for site in config.SITES:
            comp_status = "COMPLIANT"
            color = "#28a745"
            for k, v in env_results.get('exceedance', {}).items():
                if v['site'] == site:
                    if v['gt_70'] > 0:
                        comp_status = "NON-COMPLIANT"
                        color = "#dc3545"
                        break
                    elif v['gt_4'] > 0:
                        comp_status = "ADVISORY"
                        color = "#ffc107"
            env_html += f"<div class='card' style='border-top: 4px solid {color};'><h3>{site}</h3><p style='font-size: 20px; font-weight: bold; color: {color};'>{comp_status}</p></div>"
        env_html += "</div>"
        
        # Narrative
        env_html += "<h3>Environmental Interpretation</h3>"
        env_html += f"<blockquote style='background: #f9f9f9; border-left: 10px solid #ccc; margin: 1.5em 10px; padding: 0.5em 10px; white-space: pre-wrap;'>{env_results.get('narrative', '')}</blockquote>"
        
        html_content += env_html
        
    html_content += """
            <h2>Visualizations</h2>
    """
    
    plots = [
        ('Timeseries Overview', 'pfas_concentration_timeseries.png'),
        ('Predicted vs Actual', 'predicted_vs_actual.png'),
        ('72-Hour Forecast', '72hr_forecast.png'),
        ('Anomaly Heatmap', 'anomaly_heatmap.png'),
        ('Model Residuals', 'residual_plot.png'),
        ('System Scalability', 'scalability_plot.png'),
        ('Seasonal Decomposition', 'seasonal_decomposition.png'),
        ('Correlation Heatmap', 'correlation_heatmap.png'),
        ('Spatial Risk Comparison', 'spatial_risk_comparison.png')
    ]
    
    for title, filename in plots:
        img_b64 = get_base64_img(filename)
        if img_b64:
            html_content += f"""
            <div class="img-container">
                <h3>{title}</h3>
                <img src="{img_b64}" alt="{title}">
            </div>
            """
            
    html_content += f"""
            <div class="footer">
                <p>PFAS Monitor v2.0 | EPA Limit: {config.EPA_LIMIT_PPT} ppt | EPA Advisory: {config.EPA_ADVISORY_PPT} ppt</p>
            </div>
        </div>
    </body>
    </html>
    """
    
    report_path = os.path.join(output_dir, config.HTML_REPORT_FILENAME)
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
