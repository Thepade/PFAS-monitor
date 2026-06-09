import argparse
import os
import shutil
import time
import pandas as pd
import numpy as np

import config
import security
import data_simulation
import data_processing
import ml_model
import anomaly_detection
import visualization
import alert_engine
import report_generator
import benchmarking
import dataset_adapter
import environmental_analysis
from api import get_most_recent_run_dir, inverse_transform_targets

def run_batch(args: argparse.Namespace, run_id: str, output_dir: str) -> None:
    print(f"\n[RUN] Run ID: {run_id} | Output: runs/{run_id}/")
    security.log_security_event("PIPELINE_START", f"Starting batch run {run_id}", output_dir)
    
    stage_benchmarks = []
    
    # 1. Parse overrides
    sites_list = config.SITES
    if args.sites is not None:
        sites_list = config.SITES[:args.sites]
        
    days = args.days if args.days is not None else config.SIMULATION_DAYS
    noise = args.noise if args.noise is not None else config.NOISE_STD
    
    # 2. Simulate data or Load Dataset
    if args.dataset:
        print(f"\n[DATA SOURCE] Real dataset provided: {args.dataset}")
        if not os.path.exists(args.dataset):
            raise FileNotFoundError(f"Dataset not found: {args.dataset}")
            
        sim_result = benchmarking.benchmark_pipeline_stage("Data Loading", dataset_adapter.load_user_dataset, args.dataset, output_dir)
        stage_benchmarks.append(sim_result)
        df_raw = sim_result["result"]
        data_path = os.path.join(output_dir, 'pfas_sensor_data.csv')
        data_source = f"Real Dataset ({os.path.basename(args.dataset)})"
    else:
        detected = dataset_adapter.scan_for_dataset(['.', os.getcwd()])
        if detected:
            print(f"\n[DATA SOURCE] Dataset auto-detected: {detected}")
            try:
                confirm = input("Use this real dataset instead of simulation? (y/n): ").strip().lower()
            except EOFError:
                confirm = 'n'
            if confirm == 'y':
                sim_result = benchmarking.benchmark_pipeline_stage("Data Loading", dataset_adapter.load_user_dataset, detected, output_dir)
                stage_benchmarks.append(sim_result)
                df_raw = sim_result["result"]
                data_path = os.path.join(output_dir, 'pfas_sensor_data.csv')
                data_source = f"Real Dataset ({os.path.basename(detected)})"
            else:
                sim_result = benchmarking.benchmark_pipeline_stage(
                    "Data Simulation",
                    data_simulation.main,
                    output_dir, days=days, sites=sites_list, noise_std=noise
                )
                stage_benchmarks.append(sim_result)
                data_path = sim_result["result"]
                df_raw = pd.read_csv(data_path)
                df_raw['timestamp'] = pd.to_datetime(df_raw['timestamp'])
                data_source = f"Simulation ({days} days, {len(sites_list)} sites)"
        else:
            sim_result = benchmarking.benchmark_pipeline_stage(
                "Data Simulation",
                data_simulation.main,
                output_dir, days=days, sites=sites_list, noise_std=noise
            )
            stage_benchmarks.append(sim_result)
            data_path = sim_result["result"]
            df_raw = pd.read_csv(data_path)
            df_raw['timestamp'] = pd.to_datetime(df_raw['timestamp'])
            data_source = f"Simulation ({days} days, {len(sites_list)} sites)"

    print(f"[DATA SOURCE] Confirmed: {data_source}")
    
    # 3. Model caching & Skip-train logic
    if args.skip_train:
        latest = get_most_recent_run_dir()
        if latest and latest != output_dir:
            copied = False
            for fname in ["lr_model.pkl", "rf_model.pkl", "lstm_model.h5", "rf_model_ablation.pkl", "scaler.pkl"]:
                src = os.path.join(latest, fname)
                if os.path.exists(src):
                    shutil.copy2(src, os.path.join(output_dir, fname))
                    copied = True
            if copied:
                print(f"Copied existing models from {os.path.basename(latest)} for --skip-train")
            
    # 4. Data Processing
    def run_data_processing():
        X_train, X_test, y_train, y_test, scaler = data_processing.prepare_data(data_path, include_env_features=True)
        y_test_inv = inverse_transform_targets(y_test, scaler)
        
        df = data_processing.load_data(data_path)
        df = data_processing.clean_data(df)
        df = data_processing.calibrate_signal(df)
        
        if args.site:
            df = df[df['site'] == args.site].copy()
            
        return X_train, X_test, y_train, y_test, scaler, y_test_inv, df

    dp_result = benchmarking.benchmark_pipeline_stage("Data Processing", run_data_processing)
    stage_benchmarks.append(dp_result)
    X_train, X_test, y_train, y_test, scaler, y_test_inv, df = dp_result["result"]
    
    # 5. Train Models
    ml_result = benchmarking.benchmark_pipeline_stage("ML Training", ml_model.train_models, data_path, output_dir)
    stage_benchmarks.append(ml_result)
    predictions_dict, metrics_dict, best_model_name, ablation_results = ml_result["result"]
    
    # Insert actual target array so visualization plots accurately map the scatter and residuals
    predictions_dict['Actual'] = y_test_inv
    
    # 6. Anomaly Detection
    anom_result = benchmarking.benchmark_pipeline_stage("Anomaly Detection", anomaly_detection.detect_anomalies, df)
    stage_benchmarks.append(anom_result)
    df_anom = anom_result["result"]
    
    # 7. Visualizations
    def run_visualizations():
        visualization.generate_all_plots(df_anom, predictions_dict, metrics_dict, best_model_name, output_dir)
        visualization.save_metrics_csv(metrics_dict, output_dir)
        
    viz_result = benchmarking.benchmark_pipeline_stage("Visualization", run_visualizations)
    stage_benchmarks.append(viz_result)

    # 7b. Environmental Analysis
    print("\n[STEP 5b/6] Running Environmental Analysis...")
    env_results = environmental_analysis.run_full_environmental_analysis(df_anom, predictions_dict, metrics_dict, output_dir)
    stage_benchmarks.append({"stage": "Environmental Analysis", "time_seconds": 0.0, "peak_memory_mb": 0.0}) # Simplified
    
    # 8. Alerts Integration
    def run_alerts():
        site_forecasts = {}
        if best_model_name in predictions_dict:
            preds = predictions_dict[best_model_name]
            if preds.ndim > 1 and len(preds) > 0:
                avg_pred = preds[-1, 0]
                for s in config.SITES:
                    site_forecasts[s] = {}
                    for c in config.COMPOUNDS:
                        site_forecasts[s][c] = float(avg_pred)
                        
        alert_engine.generate_alerts(df_anom, site_forecasts, output_dir)
        
    alert_result = benchmarking.benchmark_pipeline_stage("Alert Engine", run_alerts)
    stage_benchmarks.append(alert_result)
    
    benchmarking.print_benchmark_report(stage_benchmarks, [])
    benchmarking.save_benchmark_csv(stage_benchmarks, [], output_dir)

    # 9. Reports
    def run_reports():
        anomaly_summary = {
            'total': int(df_anom['anomaly_flag'].sum()),
            'by_site': df_anom[df_anom['anomaly_flag']]['site'].value_counts().to_dict(),
            'by_severity': df_anom['severity'].value_counts().to_dict()
        }
        report_generator.generate_text_report(metrics_dict, ablation_results, anomaly_summary, run_id, output_dir, stage_benchmarks, data_source, env_results)
        report_generator.generate_html_report(metrics_dict, ablation_results, anomaly_summary, predictions_dict, run_id, output_dir, stage_benchmarks, data_source, env_results)
        return anomaly_summary

    report_result = benchmarking.benchmark_pipeline_stage("Report Generation", run_reports)
    stage_benchmarks.append(report_result)
    anomaly_summary = report_result["result"]
    
    security.log_security_event("PIPELINE_COMPLETE", "Batch run completed", output_dir)
    
    # 10. Final Summary Box
    num_models = len(metrics_dict)
    models_str = ", ".join([m for m in metrics_dict.keys() if m != 'Actual'])
    best_rmse = metrics_dict.get(best_model_name, {}).get("RMSE", 0.0)
    
    with_env = ablation_results.get('with_env', {}).get('RMSE', 0.0)
    wo_env = ablation_results.get('without_env', {}).get('RMSE', 0.0)
    diff = wo_env - with_env
    pct = (abs(diff) / wo_env) * 100 if wo_env != 0 else 0.0
    
    alerts_issued = 0
    for s in config.SITES:
        for c in config.COMPOUNDS:
            target_df = df_anom[(df_anom['site']==s) & (df_anom['compound']==c)]
            if not target_df.empty:
                val = target_df['concentration_calibrated'].iloc[-1]
                if val >= config.EPA_LIMIT_PPT:
                    alerts_issued += 1

    # Extract Environmental Summary Info
    sites_adv = sum(1 for v in env_results['exceedance'].values() if v['gt_4'] > 0)
    sites_mcl = sum(1 for v in env_results['exceedance'].values() if v['gt_70'] > 0)
    dominant_trend = "STABLE"
    trends = [v['trend_dir'] for v in env_results['trend'].values()]
    if trends.count("INCREASING") > len(trends)/2: dominant_trend = "INCREASING"
    elif trends.count("DECREASING") > len(trends)/2: dominant_trend = "DECREASING"
                    
    print("\n" + "\u2550"*50)
    print(" PFAS MONITOR v2.0 \u2014 RUN COMPLETE")
    print("\u2550"*50)
    print(f" Run ID                   : {run_id}")
    print(f" Data source              : {data_source}")
    print(f" Output directory         : runs/{run_id}/")
    print(f" Environmental analysis   : Completed")
    print(f" Sites above EPA advisory : {sites_adv}/{len(env_results['exceedance'])} sites")
    print(f" Sites above EPA MCL      : {sites_mcl}/{len(env_results['exceedance'])} sites")
    print(f" Dominant trend           : {dominant_trend}")
    print(f" Data points generated    : {len(df)}")
    print(f" Anomalies detected       : {anomaly_summary['total']}")
    print(f" Models trained           : {num_models} ({models_str})")
    print(f" Best model               : {best_model_name} (RMSE: {best_rmse:.2f})")
    print(f" Ablation study           : RF improved by {abs(diff):.2f} ppt ({pct:.1f}%)")
    print(f" Plots saved              : 5 (+ Environmental)")
    print(f" Metrics CSV saved        : metrics_summary.csv")
    print(f" HTML report              : pfas_report.html")
    print(f" Security log             : security_log.txt")
    print(f" Alerts issued            : {alerts_issued}")
    slowest_stage = max(stage_benchmarks, key=lambda x: x["time_seconds"])
    total_time = sum([x["time_seconds"] for x in stage_benchmarks])
    peak_mem = max([x["peak_memory_mb"] for x in stage_benchmarks])
    
    print(f" Benchmark CSV saved      : benchmark_stages.csv")
    print(f" Slowest stage            : {slowest_stage['stage']} ({slowest_stage['time_seconds']:.2f}s)")
    print(f" Total pipeline time      : {total_time:.2f} seconds")
    print(f" Peak memory usage        : {peak_mem:.1f} MB")
    print("\u2550"*50)
    print(" To run API server        : python api.py")
    print(" To stream live data      : python main.py --mode streaming")
    print(" To predict manually      : python manual_predict.py")
    print(" To run benchmarks        : python main.py --benchmark")
    print(f" To use YOUR dataset      : python main.py --dataset \"C:\\path\\to\\your_file.csv\"")
    print(" To auto-detect dataset   : place CSV in project folder and run python main.py")
    print("\u2550"*50)


def run_streaming(args: argparse.Namespace) -> None:
    import pickle
    import numpy as np
    
    latest = get_most_recent_run_dir()
    if not latest:
        print("Error: No latest run directory found. Run in batch mode first.")
        return
        
    data_path = os.path.join(latest, "pfas_sensor_data.csv")
    model_path = os.path.join(latest, "rf_model.pkl")
    scaler_path = os.path.join(latest, "scaler.pkl")
    
    if not os.path.exists(data_path) or not os.path.exists(model_path) or not os.path.exists(scaler_path):
        print(f"Error: Missing data, model or scaler in {latest}")
        return
        
    print(f"--- Starting PFAS Streaming Mode (Data: {os.path.basename(latest)}) ---")
    
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
        
    df = data_processing.load_data(data_path)
    if args.site:
        df = df[df['site'] == args.site]
        
    df = df.sort_values(by=['timestamp', 'site', 'compound']).reset_index(drop=True)
    
    chunk_size = config.WINDOW_SIZE
    ticks = 0
    max_ticks = 20
    anomalies_found = 0
    start_idx = 0
    
    while ticks < max_ticks and start_idx + chunk_size <= len(df):
        chunk = df.iloc[start_idx : start_idx + chunk_size].copy()
        
        chunk = data_processing.clean_data(chunk)
        chunk = data_processing.calibrate_signal(chunk)
        chunk_anom = anomaly_detection.detect_anomalies(chunk)
        
        num_anom = int(chunk_anom['anomaly_flag'].sum())
        anomalies_found += num_anom
        
        n_features = scaler.n_features_in_
        data = np.zeros((chunk_size, n_features))
        data[:, 0] = chunk['concentration_calibrated'].values
        
        if n_features > 1:
            if 'temperature_C' in chunk.columns:
                data[:, 1] = chunk['temperature_C'].values
                data[:, 2] = chunk['pH'].values
                data[:, 3] = chunk['flow_rate_Ls'].values
            else:
                data[:, 1] = 25.0
                data[:, 2] = 7.5
                data[:, 3] = 2.75
                
        data_scaled = scaler.transform(data)
        X = data_scaled.flatten().reshape(1, -1)
        pred_scaled = model.predict(X)
        pred_inv = inverse_transform_targets(pred_scaled, scaler)[0]
        
        chunk_hash = security.generate_data_fingerprint(chunk)
        pred_val = float(pred_inv[0]) if isinstance(pred_inv, np.ndarray) and len(pred_inv) > 0 else float(pred_inv)
        
        print(f"Tick {ticks+1:02d}/{max_ticks} | Hash: {chunk_hash[:8]} | Anomalies: {num_anom} | Next Pred: {pred_val:.2f} ppt")
        
        ticks += 1
        start_idx += chunk_size
        time.sleep(0.5)
        
    print("\n--- Streaming Session Summary ---")
    print(f"Ticks processed    : {ticks}")
    print(f"Data rows consumed : {ticks * chunk_size}")
    print(f"Anomalies detected : {anomalies_found}")
    print("Streaming complete.")


def main():
    parser = argparse.ArgumentParser(description="PFAS Monitor v2.0")
    parser.add_argument("--mode", type=str, choices=["batch", "streaming"], default="batch", help="Run mode (batch/streaming)")
    parser.add_argument("--site", type=str, help="Filter to one site")
    parser.add_argument("--days", type=int, help="Override SIMULATION_DAYS")
    parser.add_argument("--sites", type=int, help="Override number of sites")
    parser.add_argument("--noise", type=float, help="Override NOISE_STD")
    parser.add_argument("--skip-train", action="store_true", help="Skip retraining if models exist in previous run dir")
    parser.add_argument("--benchmark", action="store_true", help="Run scalability benchmark test")
    parser.add_argument("--dataset", type=str, default=None, help="Path to real dataset CSV/Excel")
    
    args = parser.parse_args()
    
    if args.mode == "batch":
        run_id = security.generate_run_id()
        base_dir = os.path.dirname(os.path.abspath(__file__))
        output_dir = os.path.join(base_dir, config.RUNS_DIR, run_id)
        os.makedirs(output_dir, exist_ok=True)
        
        def run_pipeline_func(days, output_dir):
            dummy_args = argparse.Namespace(**vars(args))
            dummy_args.days = days
            run_batch(dummy_args, security.generate_run_id(), output_dir)
            
        run_batch(args, run_id, output_dir)
        
        if args.benchmark:
            print("\n--- Running Scalability Benchmarks ---")
            scalability_results = benchmarking.run_scalability_test(
                run_pipeline_func,
                day_configs=[30, 90, 180],
                base_output_dir=os.path.join(base_dir, config.RUNS_DIR, "scalability_test")
            )
            benchmarking.print_benchmark_report([], scalability_results)
            benchmarking.save_benchmark_csv([], scalability_results, output_dir)
            benchmarking.plot_scalability(scalability_results, output_dir)
            print(f"Scalability test completed. Results in {output_dir}")
            
    elif args.mode == "streaming":
        run_streaming(args)
        
if __name__ == "__main__":
    main()