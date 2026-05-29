#!/usr/bin/env python3
"""Fetch GPU utilization from wandb runs and export organized results."""

import wandb
import json
import csv
import os
from datetime import datetime

# Run data: (run_name, run_id, fold, nw, pf)
RUNS = [
    ("fbirn_multimodal_wp_nw8_pf8", "cx9n5epa", 1, 8, 8),
    ("fbirn_multimodal_wp_nw8_pf4", "38xj35it", 1, 8, 4),
    ("fbirn_multimodal_wp_nw8_pf8", "chm5qqtx", 0, 8, 8),
    ("fbirn_multimodal_wp_nw8_pf4", "j0ae4scj", 0, 8, 4),
    ("fbirn_multimodal_wp_nw8_pf2", "ciixul4b", 0, 8, 2),
    ("fbirn_multimodal_wp_nw2_pf2", "3tw2n2ga", 1, 2, 2),
    ("fbirn_multimodal_wp_nw2_pf4", "irek2yyr", 1, 2, 4),
    ("fbirn_multimodal_wp_nw2_pf8", "whtg9aru", 1, 2, 8),
    ("fbirn_multimodal_wp_nw4_pf8", "hdhdjmsx", 1, 4, 8),
    ("fbirn_multimodal_wp_nw4_pf2", "krt1puf5", 1, 4, 2),
    ("fbirn_multimodal_wp_nw4_pf4", "0rkrodia", 1, 4, 4),
    ("fbirn_multimodal_dense_e10", "hcet6mdp", 1, 4, 4),
    ("fbirn_multimodal_wp_nw2_pf4", "n1qd6te9", 0, 2, 4),
    ("fbirn_multimodal_wp_nw4_pf8", "v3aea182", 0, 4, 8),
    ("fbirn_multimodal_wp_nw2_pf2", "at3kwnk5", 0, 2, 2),
    ("fbirn_multimodal_wp_nw2_pf8", "be53jye2", 0, 2, 8),
    ("fbirn_multimodal_wp_nw4_pf4", "k7b6nq68", 0, 4, 4),
    ("fbirn_multimodal_wp_nw4_pf2", "fjzzv8ba", 0, 4, 2),
    ("fbirn_multimodal_dense_e10", "eavbj9ew", 0, 4, 4),
]

ENTITY = "neuroneural"
PROJECT = "holographic_project"
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "runs-results")

def fetch_gpu_metrics():
    api = wandb.Api()
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    results = []
    
    for run_name, run_id, fold, nw, pf in RUNS:
        print(f"Fetching: {run_name} (fold={fold}, id={run_id})")
        try:
            run = api.run(f"{ENTITY}/{PROJECT}/{run_id}")
            
            # Get GPU metrics from system history
            history = run.history(stream="system")
            
            # Look for ALL gpu-related columns
            gpu_cols = [c for c in history.columns if 'gpu' in c.lower()]
            
            gpu_utils = []
            for col in gpu_cols:
                vals = history[col].dropna().tolist()
                if vals:
                    avg = sum(vals) / len(vals)
                    gpu_utils.append({
                        "metric": col,
                        "avg": round(avg, 2),
                        "min": round(min(vals), 2),
                        "max": round(max(vals), 2),
                        "samples": len(vals)
                    })
            
            # Also check summary for logged gpu metrics
            summary_gpu = {k: v for k, v in run.summary.items() if 'gpu' in k.lower()}
            
            # Get run timestamps
            created = run.created_at
            timestamp = run.summary.get("_timestamp", None)
            
            results.append({
                "run_id": run_id,
                "run_name": run_name,
                "fold": fold,
                "nw": nw,
                "pf": pf,
                "config": f"nw={nw} pf={pf}",
                "gpu_metrics": gpu_utils,
                "summary_gpu": summary_gpu,
                "created_at": str(created),
                "state": run.state,
            })
            
            if gpu_utils:
                print(f"  GPU metrics: {[(g['metric'], g['avg']) for g in gpu_utils]}")
            else:
                print(f"  No GPU metrics found. Checking for logged metrics...")
                # Check if gpu_utilization was logged as a custom metric
                hist = run.history()
                gpu_logged = [c for c in hist.columns if 'gpu' in c.lower()]
                print(f"  Logged GPU columns: {gpu_logged}")
                if gpu_logged:
                    for col in gpu_logged:
                        vals = hist[col].dropna().tolist()
                        if vals:
                            avg = sum(vals) / len(vals)
                            gpu_utils.append({
                                "metric": col,
                                "avg": round(avg, 2),
                                "min": round(min(vals), 2),
                                "max": round(max(vals), 2),
                                "samples": len(vals)
                            })
                            print(f"  Found logged metric {col}: avg={avg}")
            
        except Exception as e:
            print(f"  Error: {e}")
            results.append({
                "run_id": run_id,
                "run_name": run_name,
                "fold": fold,
                "nw": nw,
                "pf": pf,
                "config": f"nw={nw} pf={pf}",
                "error": str(e),
            })
    
    return results


def export_results(results):
    # Export JSON
    json_path = os.path.join(OUTPUT_DIR, "gpu_utilization.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nJSON exported: {json_path}")
    
    # Export CSV summary
    csv_path = os.path.join(OUTPUT_DIR, "gpu_utilization_summary.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "run_id", "run_name", "fold", "nw", "pf", "config",
            "gpu_metric", "avg_util", "min_util", "max_util", "samples",
            "created_at", "state"
        ])
        
        for r in results:
            if "error" in r:
                writer.writerow([
                    r["run_id"], r["run_name"], r["fold"], r["nw"], r["pf"],
                    r["config"], "ERROR", r["error"], "", "", "", r.get("created_at", ""), r.get("state", "")
                ])
                continue
            
            if not r["gpu_metrics"]:
                writer.writerow([
                    r["run_id"], r["run_name"], r["fold"], r["nw"], r["pf"],
                    r["config"], "NO_GPU_DATA", "", "", "", "",
                    r.get("created_at", ""), r.get("state", "")
                ])
                continue
            
            for g in r["gpu_metrics"]:
                writer.writerow([
                    r["run_id"], r["run_name"], r["fold"], r["nw"], r["pf"],
                    r["config"], g["metric"], g["avg"], g["min"], g["max"],
                    g["samples"], r.get("created_at", ""), r.get("state", "")
                ])
    
    print(f"CSV exported: {csv_path}")
    
    # Print comparison table
    print("\n" + "="*90)
    print(f"{'Run ID':<12} {'Config':<14} {'Fold':<5} {'Metric':<30} {'Avg':<10} {'Min':<10} {'Max':<10}")
    print("="*90)
    for r in results:
        if "error" in r or not r.get("gpu_metrics"):
            print(f"{r['run_id']:<12} {r['config']:<14} {r['fold']:<5} {'NO_GPU_DATA':<30}")
            continue
        for g in r["gpu_metrics"]:
            print(f"{r['run_id']:<12} {r['config']:<14} {r['fold']:<5} {g['metric']:<30} {g['avg']:<10} {g['min']:<10} {g['max']:<10}")
    print("="*90)


if __name__ == "__main__":
    results = fetch_gpu_metrics()
    export_results(results)
