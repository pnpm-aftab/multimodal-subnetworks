#!/usr/bin/env python3
"""Generate a clean GPU utilization comparison summary from fetched results."""

import json
import csv
import os

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "runs-results")

def generate_summary():
    with open(os.path.join(RESULTS_DIR, "gpu_utilization.json")) as f:
        results = json.load(f)
    
    # Extract GPU utilization (system.gpu.X.gpu) for each run
    print("="*100)
    print(f"{'Run ID':<12} {'Config':<14} {'Fold':<5} {'GPU':<8} {'Avg %':<10} {'Min %':<10} {'Max %':<10} {'Avg Mem%':<10} {'Avg Alloc%':<12} {'Avg Power%':<12}")
    print("="*100)
    
    # CSV for utilization summary
    csv_path = os.path.join(RESULTS_DIR, "gpu_utilization_comparison.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["run_id", "run_name", "fold", "nw", "pf", "gpu_id", "avg_util", "min_util", "max_util", "avg_memory", "avg_memory_allocated", "avg_power_percent"])
        
        for r in results:
            run_id = r["run_id"]
            config = r["config"]
            fold = r["fold"]
            nw = r["nw"]
            pf = r["pf"]
            
            if "error" in r or not r.get("gpu_metrics"):
                print(f"{run_id:<12} {config:<14} {fold:<5} {'NO DATA':<50}")
                writer.writerow([run_id, r["run_name"], fold, nw, pf, "N/A", "N/A", "N/A", "N/A", "N/A", "N/A", "N/A"])
                continue
            
            # Group by GPU
            gpu_metrics = r["gpu_metrics"]
            gpu_ids = set()
            for g in gpu_metrics:
                # Extract gpu id from metric name like system.gpu.0.gpu
                parts = g["metric"].split(".")
                if len(parts) >= 4 and parts[1] == "gpu":
                    gpu_ids.add(f"{parts[1]}.{parts[2]}")  # e.g. "gpu.0"
            
            for gpu_id in sorted(gpu_ids):
                # gpu_id is like "gpu.0", metric is "system.gpu.0.gpu"
                util_metric = f"system.{gpu_id}.gpu"
                mem_metric = f"system.{gpu_id}.memory"
                mem_alloc_metric = f"system.{gpu_id}.memoryAllocated"
                power_metric = f"system.{gpu_id}.powerPercent"
                
                util = next((g for g in gpu_metrics if g["metric"] == util_metric), None)
                mem = next((g for g in gpu_metrics if g["metric"] == mem_metric), None)
                mem_alloc = next((g for g in gpu_metrics if g["metric"] == mem_alloc_metric), None)
                power = next((g for g in gpu_metrics if g["metric"] == power_metric), None)
                
                avg_util = util["avg"] if util else "N/A"
                min_util = util["min"] if util else "N/A"
                max_util = util["max"] if util else "N/A"
                avg_mem = mem["avg"] if mem else "N/A"
                avg_mem_alloc = mem_alloc["avg"] if mem_alloc else "N/A"
                avg_power = power["avg"] if power else "N/A"
                
                gpu_num = gpu_id.split(".")[1]  # "gpu.0" -> "0"
                print(f"{run_id:<12} {config:<14} {fold:<5} {gpu_num:<8} {avg_util:<10} {min_util:<10} {max_util:<10} {avg_mem:<10} {avg_mem_alloc:<12} {avg_power:<12}")
                
                writer.writerow([run_id, r["run_name"], fold, nw, pf, gpu_num, avg_util, min_util, max_util, avg_mem, avg_mem_alloc, avg_power])
    
    print("="*100)
    print(f"\nCSV summary exported: {csv_path}")
    
    # Aggregate by config (average across folds and GPUs)
    print("\n\nAGGREGATE BY CONFIG (avg across folds & GPUs):")
    print("="*80)
    print(f"{'Config':<14} {'Runs':<6} {'Avg GPU %':<12} {'Min GPU %':<12} {'Max GPU %':<12}")
    print("="*80)
    
    config_stats = {}
    for r in results:
        config = r["config"]
        if "error" in r or not r.get("gpu_metrics"):
            continue
        
        for g in r["gpu_metrics"]:
            if g["metric"].endswith(".gpu"):
                if config not in config_stats:
                    config_stats[config] = {"utils": [], "mins": [], "maxs": []}
                config_stats[config]["utils"].append(g["avg"])
                config_stats[config]["mins"].append(g["min"])
                config_stats[config]["maxs"].append(g["max"])
    
    # Also aggregate by nw value
    nw_stats = {}
    for config, stats in config_stats.items():
        nw = int(config.split("nw=")[1].split()[0])
        if nw not in nw_stats:
            nw_stats[nw] = {"utils": [], "mins": [], "maxs": [], "configs": 0}
        nw_stats[nw]["utils"].extend(stats["utils"])
        nw_stats[nw]["mins"].extend(stats["mins"])
        nw_stats[nw]["maxs"].extend(stats["maxs"])
        nw_stats[nw]["configs"] += len(stats["utils"])
    
    for config in sorted(config_stats.keys()):
        stats = config_stats[config]
        avg = round(sum(stats["utils"]) / len(stats["utils"]), 2)
        min_v = round(min(stats["mins"]), 2)
        max_v = round(max(stats["maxs"]), 2)
        print(f"{config:<14} {len(stats['utils']):<6} {avg:<12} {min_v:<12} {max_v:<12}")
    
    print("="*80)
    
    print("\n\nAGGREGATE BY NUM WORKERS (nw):")
    print("="*70)
    print(f"{'nw':<6} {'Samples':<10} {'Avg GPU %':<12} {'Min GPU %':<12} {'Max GPU %':<12}")
    print("="*70)
    for nw in sorted(nw_stats.keys()):
        stats = nw_stats[nw]
        avg = round(sum(stats["utils"]) / len(stats["utils"]), 2)
        min_v = round(min(stats["mins"]), 2)
        max_v = round(max(stats["maxs"]), 2)
        print(f"{nw:<6} {stats['configs']:<10} {avg:<12} {min_v:<12} {max_v:<12}")
    print("="*70)

    # Export aggregate CSV
    agg_csv = os.path.join(RESULTS_DIR, "gpu_utilization_aggregate.csv")
    with open(agg_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["group_by", "value", "samples", "avg_gpu_pct", "min_gpu_pct", "max_gpu_pct"])
        for config in sorted(config_stats.keys()):
            stats = config_stats[config]
            avg = round(sum(stats["utils"]) / len(stats["utils"]), 2)
            min_v = round(min(stats["mins"]), 2)
            max_v = round(max(stats["maxs"]), 2)
            writer.writerow(["config", config, len(stats["utils"]), avg, min_v, max_v])
        
        for nw in sorted(nw_stats.keys()):
            stats = nw_stats[nw]
            avg = round(sum(stats["utils"]) / len(stats["utils"]), 2)
            min_v = round(min(stats["mins"]), 2)
            max_v = round(max(stats["maxs"]), 2)
            writer.writerow(["nw", nw, stats["configs"], avg, min_v, max_v])
    
    print(f"\nAggregate CSV exported: {agg_csv}")


if __name__ == "__main__":
    generate_summary()
