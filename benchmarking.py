import time
import tracemalloc
import os
import csv
import matplotlib.pyplot as plt
from typing import Dict, Callable, Any
import config

def time_function(func: Callable, *args, **kwargs) -> tuple:
    start_time = time.perf_counter()
    result = func(*args, **kwargs)
    end_time = time.perf_counter()
    elapsed = end_time - start_time
    return result, elapsed

def memory_profile(func: Callable, *args, **kwargs) -> tuple:
    tracemalloc.start()
    result = func(*args, **kwargs)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    peak_mb = peak / (1024 * 1024)
    return result, peak_mb

def benchmark_pipeline_stage(stage_name: str, func: Callable, *args, **kwargs) -> Dict:
    def wrapped(*args, **kwargs):
        # We need both time and memory. Best to run them concurrently in a single pass if possible,
        # but tracemalloc wraps the whole execution.
        start_time = time.perf_counter()
        tracemalloc.start()
        
        result = func(*args, **kwargs)
        
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        end_time = time.perf_counter()
        
        return result, (end_time - start_time), (peak / (1024 * 1024))
        
    result, elapsed, peak_mb = wrapped(*args, **kwargs)
    
    return {
        "stage": stage_name, 
        "time_seconds": elapsed, 
        "peak_memory_mb": peak_mb,
        "result": result
    }

def run_scalability_test(main_func: Callable, day_configs: list, base_output_dir: str) -> list:
    results = []
    for days in day_configs:
        out_dir = os.path.join(base_output_dir, f"scalability_run_{days}d")
        os.makedirs(out_dir, exist_ok=True)
        
        def run_target():
            return main_func(days=days, output_dir=out_dir)
            
        _, elapsed = time_function(run_target)
        
        # We need to run memory profile separately or just rely on a single run to get both
        # Let's run it once with both tracking
        def run_both():
            start_time = time.perf_counter()
            tracemalloc.start()
            main_func(days=days, output_dir=out_dir)
            _, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            end_time = time.perf_counter()
            return (end_time - start_time), (peak / (1024 * 1024))
            
        elapsed, peak_mb = run_both()
        
        # To get data points we can calculate or just record days
        results.append({
            "days": days,
            "data_points": days * 24 * len(config.SITES) * len(config.COMPOUNDS),
            "time_seconds": elapsed,
            "peak_memory_mb": peak_mb
        })
    return results

def print_benchmark_report(stage_results: list, scalability_results: list) -> None:
    print("\n\u250c" + "\u2500"*29 + "\u252c" + "\u2500"*14 + "\u252c" + "\u2500"*18 + "\u2510")
    print("\u2502 Pipeline Stage              \u2502 Time (sec)   \u2502 Peak Memory (MB) \u2502")
    print("\u251c" + "\u2500"*29 + "\u253c" + "\u2500"*14 + "\u253c" + "\u2500"*18 + "\u2524")
    
    total_time = 0.0
    total_mem = 0.0
    for r in stage_results:
        print(f"\u2502 {r['stage']:<27} \u2502 {r['time_seconds']:12.2f} \u2502 {r['peak_memory_mb']:16.1f} \u2502")
        total_time += r['time_seconds']
        total_mem = max(total_mem, r['peak_memory_mb']) # Use max for peak memory across stages
        
    print("\u2502 " + "\u2500"*27 + " \u2502 " + "\u2500"*12 + " \u2502 " + "\u2500"*16 + " \u2502")
    print(f"\u2502 {'TOTAL PIPELINE':<27} \u2502 {total_time:12.2f} \u2502 {total_mem:16.1f} \u2502")
    print("\u2514" + "\u2500"*29 + "\u2534" + "\u2500"*14 + "\u2534" + "\u2500"*18 + "\u2518")
    
    if scalability_results:
        print("\n\u250c" + "\u2500"*12 + "\u252c" + "\u2500"*14 + "\u252c" + "\u2500"*14 + "\u252c" + "\u2500"*18 + "\u2510")
        print("\u2502 Days       \u2502 Data Points  \u2502 Time (sec)   \u2502 Peak Memory (MB) \u2502")
        print("\u251c" + "\u2500"*12 + "\u253c" + "\u2500"*14 + "\u253c" + "\u2500"*14 + "\u253c" + "\u2500"*18 + "\u2524")
        for r in scalability_results:
            print(f"\u2502 {r['days']:<10} \u2502 {r['data_points']:<12} \u2502 {r['time_seconds']:12.2f} \u2502 {r['peak_memory_mb']:16.1f} \u2502")
        print("\u2514" + "\u2500"*12 + "\u2534" + "\u2500"*14 + "\u2534" + "\u2500"*14 + "\u2534" + "\u2500"*18 + "\u2518")

def save_benchmark_csv(stage_results: list, scalability_results: list, output_dir: str) -> None:
    stages_path = os.path.join(output_dir, "benchmark_stages.csv")
    with open(stages_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Stage", "Time (sec)", "Peak Memory (MB)"])
        for r in stage_results:
            writer.writerow([r['stage'], f"{r['time_seconds']:.2f}", f"{r['peak_memory_mb']:.1f}"])
            
    if scalability_results:
        scal_path = os.path.join(output_dir, "benchmark_scalability.csv")
        with open(scal_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Days", "Data Points", "Time (sec)", "Peak Memory (MB)"])
            for r in scalability_results:
                writer.writerow([r['days'], r['data_points'], f"{r['time_seconds']:.2f}", f"{r['peak_memory_mb']:.1f}"])

def plot_scalability(scalability_results: list, output_dir: str) -> None:
    if not scalability_results:
        return
        
    days = [r['days'] for r in scalability_results]
    times = [r['time_seconds'] for r in scalability_results]
    mems = [r['peak_memory_mb'] for r in scalability_results]
    
    fig, ax1 = plt.subplots(figsize=(10, 6))
    plt.style.use('seaborn-v0_8-whitegrid')
    
    color1 = 'tab:red'
    ax1.set_xlabel('Days Simulated')
    ax1.set_ylabel('Execution Time (seconds)', color=color1)
    ax1.plot(days, times, marker='o', color=color1, linewidth=2, label='Time (s)')
    ax1.tick_params(axis='y', labelcolor=color1)
    
    ax2 = ax1.twinx()
    color2 = 'tab:blue'
    ax2.set_ylabel('Peak Memory (MB)', color=color2)
    ax2.plot(days, mems, marker='s', color=color2, linewidth=2, linestyle='--', label='Memory (MB)')
    ax2.tick_params(axis='y', labelcolor=color2)
    
    plt.title("System Scalability Analysis")
    fig.tight_layout()
    
    plot_path = os.path.join(output_dir, "scalability_plot.png")
    plt.savefig(plot_path, dpi=config.FIGURE_DPI)
    plt.close()
