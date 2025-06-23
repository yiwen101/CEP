#!/usr/bin/env python3
"""
Improved script to manually create experiment_summary.json from partial results.
Better parsing logic and more robust error handling.
"""

import json
import os
import glob
import re
from typing import Dict, Any, List, Tuple


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


def parse_run_name(run_name: str) -> Tuple[str, str, str]:
    """
    Parse run name to extract domain, model, and method.
    
    Expected formats:
    - domain_model_method
    - domain_model_method_index
    - domain_model_method_category_index
    
    Returns:
        Tuple of (domain, model, method)
    """
    parts = run_name.split("_")
    
    if len(parts) < 3:
        print(f"Warning: Cannot parse run name {run_name} (too few parts)")
        return None, None, None
    
    # Look for model patterns
    model_patterns = [
        r"gpt-\d+\.\d+-turbo",
        r"gpt-\d+\.\d+",
        r"claude-\d+\.\d+",
        r"llama-\d+",
        r"qwen\d+",
        r"mistral",
        r"gemini"
    ]
    
    model_idx = None
    model_name = None
    
    # Find model by pattern matching
    for i, part in enumerate(parts):
        for pattern in model_patterns:
            if re.match(pattern, part):
                model_idx = i
                model_name = part
                break
        if model_idx is not None:
            break
    
    # If no pattern match, try heuristic approach
    if model_idx is None:
        for i, part in enumerate(parts):
            if any(keyword in part.lower() for keyword in ["gpt", "claude", "llama", "qwen", "mistral", "gemini"]):
                model_idx = i
                model_name = part
                break
    
    if model_idx is None:
        print(f"Warning: Cannot find model in run name {run_name}")
        return None, None, None
    
    # Extract components
    domain = "_".join(parts[:model_idx])
    
    # Handle multi-part model names (e.g., "gpt-3.5-turbo")
    if model_idx + 1 < len(parts) and parts[model_idx + 1] in ["turbo", "plus", "flash"]:
        model = f"{parts[model_idx]}_{parts[model_idx + 1]}"
        method_start = model_idx + 2
    else:
        model = parts[model_idx]
        method_start = model_idx + 1
    
    # Extract method (everything after model)
    if method_start < len(parts):
        method = "_".join(parts[method_start:])
    else:
        method = "unknown"
    
    return domain, model, method


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
    processed_count = 0
    skipped_count = 0
    
    for run_dir in run_dirs:
        run_name = os.path.basename(run_dir)
        print(f"Processing: {run_name}")
        
        # Parse run name
        domain, model, method = parse_run_name(run_name)
        
        if domain is None or model is None or method is None:
            print(f"  Skipping due to parsing error")
            skipped_count += 1
            continue
        
        # Load run results
        run_results = load_run_results(run_dir)
        if run_results is None:
            print(f"  Skipping due to missing results")
            skipped_count += 1
            continue
        
        # Add to summary
        if domain not in summary:
            summary[domain] = {}
        
        if model not in summary[domain]:
            summary[domain][model] = {}
        
        summary[domain][model][method] = run_results
        processed_count += 1
        print(f"  Added: {domain} -> {model} -> {method}")
    
    print(f"\nProcessing complete: {processed_count} processed, {skipped_count} skipped")
    return summary


def save_experiment_summary(experiment_dir: str, summary: Dict[str, Any]):
    """Save the experiment summary to file."""
    summary_path = os.path.join(experiment_dir, "experiment_summary.json")
    
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"Experiment summary saved to: {summary_path}")


def print_summary_statistics(summary: Dict[str, Any]):
    """Print formatted summary statistics."""
    print("\n" + "="*60)
    print("EXPERIMENT SUMMARY STATISTICS")
    print("="*60)
    
    for domain, models in summary.items():
        print(f"\nDomain: {domain}")
        print("-" * 40)
        
        for model, methods in models.items():
            print(f"  Model: {model}")
            
            # Sort methods by F1 score for better readability
            method_scores = []
            for method, results in methods.items():
                metrics = results.get("metrics", {})
                sample_count = metrics.get("sample_count", 0)
                
                if "avg_f1" in metrics:
                    f1 = metrics["avg_f1"]
                    method_scores.append((method, f1, sample_count, "F1"))
                elif "avg_is_exact_match" in metrics:
                    exact_match = metrics["avg_is_exact_match"]
                    method_scores.append((method, exact_match, sample_count, "Exact Match"))
            
            # Sort by score (descending)
            method_scores.sort(key=lambda x: x[1], reverse=True)
            
            for method, score, sample_count, metric_type in method_scores:
                print(f"    {method:30} {metric_type}={score:.3f} (n={sample_count})")
    
    print("="*60)


def main():
    """Main function to create experiment summary."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Create experiment summary from partial results")
    parser.add_argument("experiment_dir", help="Path to experiment directory")
    parser.add_argument("--output", help="Output file path (optional)")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be processed without saving")
    
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
    
    # Print statistics
    print_summary_statistics(summary)
    
    # Save summary
    if not args.dry_run:
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(summary, f, indent=2)
            print(f"\nSummary saved to: {args.output}")
        else:
            save_experiment_summary(args.experiment_dir, summary)
    else:
        print("\nDry run mode - no files saved")


if __name__ == "__main__":
    main() 