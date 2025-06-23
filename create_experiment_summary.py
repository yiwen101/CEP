#!/usr/bin/env python3
"""
Script to manually create experiment_summary.json from partial results.
Useful when an experiment is terminated halfway and the summary wasn't generated.
"""

import json
import os
import glob
from typing import Dict, Any, List


def load_run_results(run_dir: str) -> Dict[str, Any]:
    """Load results from a single run directory."""
    run_json_path = os.path.join(run_dir, "run.json")
    
    if not os.path.exists(run_json_path):
        print(f"Warning: run.json not found in {run_dir}")
        return None
    
    try:
        with open(run_json_path, 'r') as f:
            run_data = json.load(f)
        
        # Extract the aggregated results
        aggregated_results = run_data.get("aggregated_run_results", {})
        run_meta = run_data.get("run_meta", {})
        
        if not aggregated_results:
            print(f"Warning: No aggregated results found in {run_dir}")
            return None
        
        return {
            "run_id": run_meta.get("run_id", os.path.basename(run_dir)),
            "metrics": aggregated_results
        }
    
    except Exception as e:
        print(f"Error loading {run_json_path}: {e}")
        return None


def create_experiment_summary(experiment_dir: str) -> Dict[str, Any]:
    """Create experiment summary from partial results."""
    
    # Load experiment metadata
    meta_path = os.path.join(experiment_dir, "experiment_meta.json")
    if os.path.exists(meta_path):
        with open(meta_path, 'r') as f:
            experiment_meta = json.load(f)
    else:
        experiment_meta = {"experiment_id": os.path.basename(experiment_dir)}
    
    # Find all run directories
    runs_dir = os.path.join(experiment_dir, "runs")
    if not os.path.exists(runs_dir):
        print(f"Error: runs directory not found in {experiment_dir}")
        return {}
    
    run_dirs = [d for d in glob.glob(os.path.join(runs_dir, "*")) if os.path.isdir(d)]
    
    if not run_dirs:
        print(f"No run directories found in {runs_dir}")
        return {}
    
    print(f"Found {len(run_dirs)} run directories")
    
    # Process each run
    summary = {}
    
    for run_dir in run_dirs:
        run_name = os.path.basename(run_dir)
        print(f"Processing: {run_name}")
        
        # Parse run name to extract domain, model, and method
        # Format: domain_model_method or domain_model_method_index
        parts = run_name.split("_")
        
        if len(parts) < 3:
            print(f"Warning: Cannot parse run name {run_name}")
            continue
        
        # Find the model part (usually contains model name like gpt-3.5-turbo)
        model_idx = None
        for i, part in enumerate(parts):
            if "gpt" in part or "claude" in part or "llama" in part:
                model_idx = i
                break
        
        if model_idx is None:
            print(f"Warning: Cannot find model in run name {run_name}")
            continue
        
        # Extract components
        domain = "_".join(parts[:model_idx])
        model = "_".join(parts[model_idx:model_idx+2]) if model_idx + 1 < len(parts) else parts[model_idx]
        method = "_".join(parts[model_idx+2:])
        
        # Load run results
        run_results = load_run_results(run_dir)
        if run_results is None:
            continue
        
        # Add to summary
        if domain not in summary:
            summary[domain] = {}
        
        if model not in summary[domain]:
            summary[domain][model] = {}
        
        summary[domain][model][method] = run_results
    
    return summary


def save_experiment_summary(experiment_dir: str, summary: Dict[str, Any]):
    """Save the experiment summary to file."""
    summary_path = os.path.join(experiment_dir, "experiment_summary.json")
    
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"Experiment summary saved to: {summary_path}")


def main():
    """Main function to create experiment summary."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Create experiment summary from partial results")
    parser.add_argument("experiment_dir", help="Path to experiment directory")
    parser.add_argument("--output", help="Output file path (optional)")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.experiment_dir):
        print(f"Error: Experiment directory {args.experiment_dir} does not exist")
        return
    
    print(f"Creating experiment summary for: {args.experiment_dir}")
    
    # Create summary
    summary = create_experiment_summary(args.experiment_dir)
    
    if not summary:
        print("No valid results found to create summary")
        return
    
    # Save summary
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"Summary saved to: {args.output}")
    else:
        save_experiment_summary(args.experiment_dir, summary)
    
    # Print summary statistics
    print("\nSummary Statistics:")
    for domain, models in summary.items():
        print(f"\nDomain: {domain}")
        for model, methods in models.items():
            print(f"  Model: {model}")
            for method, results in methods.items():
                metrics = results.get("metrics", {})
                sample_count = metrics.get("sample_count", 0)
                if "avg_f1" in metrics:
                    f1 = metrics["avg_f1"]
                    print(f"    {method}: F1={f1:.3f}, samples={sample_count}")
                elif "avg_is_exact_match" in metrics:
                    exact_match = metrics["avg_is_exact_match"]
                    print(f"    {method}: Exact Match={exact_match:.3f}, samples={sample_count}")


if __name__ == "__main__":
    main() 