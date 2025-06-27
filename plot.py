#!/usr/bin/env python3
"""
Script to generate plots from experiment_summary.json files.
Plots recall and correctness metrics per model per domain.
"""

import json
import os
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List, Any
import argparse
from pathlib import Path
import logging
import shutil

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


def load_experiment_summary(file_path: str) -> Dict[str, Any]:
    """Load experiment summary from JSON file."""
    logging.info(f"Loading experiment summary from: {file_path}")
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
            logging.info(f"Successfully loaded data with {len(data)} domains")
            return data
    except Exception as e:
        logging.error(f"Error loading file: {str(e)}")
        raise


def get_metric_value(metrics: Dict[str, float], metric_name: str) -> float:
    """Get metric value handling different possible metric names."""
    if metric_name == 'recall':
        return metrics.get('avg_recall', metrics.get('recall', 0.0))
    elif metric_name == 'correct':
        return metrics.get('avg_correct', metrics.get('correct', 0.0))
    return metrics.get(metric_name, 0.0)

def create_bar_plot(data: Dict[str, Dict], domain: str, model: str, metric: str, output_path: str):
    """Create a bar plot for a specific metric."""
    logging.info(f"Creating bar plot for {domain} - {model} - {metric}")
    
    # Get all methods and their values
    methods = []
    values = []
    
    model_data = data[domain][model]
    for method, method_data in model_data.items():
        if 'metrics' in method_data:
            methods.append(method)
            values.append(get_metric_value(method_data['metrics'], metric))
    
    if not methods:
        logging.warning(f"No valid methods found for {domain} - {model}")
        return
    
    # Sort methods by values
    sorted_indices = np.argsort(values)[::-1]  # Sort in descending order
    methods = [methods[i] for i in sorted_indices]
    values = [values[i] for i in sorted_indices]
    
    # Create plot
    plt.figure(figsize=(15, 6))
    bars = plt.bar(range(len(methods)), values, color='#2ecc71', alpha=0.8)
    
    # Add value labels on top of bars
    for i, v in enumerate(values):
        plt.text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom', fontsize=8)
    
    # Customize plot
    metric_name = 'Recall' if metric == 'recall' else 'Correctness'
    plt.title(f'{metric_name} for {domain} - {model}', fontsize=14, pad=20)
    plt.ylabel(f'{metric_name} Score', fontsize=12)
    plt.xticks(range(len(methods)), methods, rotation=45, ha='right')
    plt.grid(True, alpha=0.3, axis='y')
    plt.ylim(0, 1.0)
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logging.info(f"Saved plot to {output_path}")


def create_summary_table(data: Dict[str, Dict], domain: str, model: str, output_path: str):
    """Create a summary table."""
    logging.info(f"Creating summary table for {domain} - {model}")
    
    # Get all methods and their metrics
    table_data = []
    model_data = data[domain][model]
    
    for method, method_data in model_data.items():
        if 'metrics' in method_data:
            metrics = method_data['metrics']
            table_data.append({
                'Method': method,
                'Recall': get_metric_value(metrics, 'recall'),
                'Correctness': get_metric_value(metrics, 'correct')
            })
    
    if not table_data:
        logging.warning(f"No valid data found for {domain} - {model}")
        return
    
    # Sort by recall score
    table_data.sort(key=lambda x: x['Recall'], reverse=True)
    
    # Create figure and table
    fig, ax = plt.subplots(figsize=(12, len(table_data) * 0.4 + 1))
    ax.axis('tight')
    ax.axis('off')
    
    # Create table
    cell_text = [[d['Method'], f"{d['Recall']:.3f}", f"{d['Correctness']:.3f}"] for d in table_data]
    table = ax.table(cellText=cell_text,
                    colLabels=['Method', 'Recall', 'Correctness'],
                    cellLoc='center',
                    loc='center')
    
    # Style table
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.5)
    
    # Color header
    for j in range(3):
        table[(0, j)].set_facecolor('#4CAF50')
        table[(0, j)].set_text_props(color='white', weight='bold')
    
    # Set title
    plt.title(f'Results Summary for {domain} - {model}\n(Sorted by Recall)', 
              pad=20, fontsize=14)
    
    # Save table
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logging.info(f"Saved table to {output_path}")


def main():
    """Main function to create plots from experiment summary."""
    parser = argparse.ArgumentParser(description='Create plots from experiment summary')
    parser.add_argument('experiment_id', nargs='?', help='Optional: Specific experiment ID to plot results for. If not provided, will plot all experiments.')
    args = parser.parse_args()
    
    # Clean up plots directory if no specific experiment
    if not args.experiment_id:
        if os.path.exists('plots'):
            logging.info("Removing existing plots directory")
            shutil.rmtree('plots')
    
    # Get list of experiments to process
    if args.experiment_id:
        experiments = [args.experiment_id]
    else:
        # Find all experiment directories in results that have experiment_summary.json
        experiments = []
        for exp_dir in os.listdir('results'):
            summary_path = os.path.join('results', exp_dir, 'experiment_summary.json')
            if os.path.exists(summary_path):
                experiments.append(exp_dir)
        logging.info(f"Found {len(experiments)} experiments to process")
    
    # Process each experiment
    for experiment_id in experiments:
        logging.info(f"Processing experiment: {experiment_id}")
        summary_path = os.path.join('results', experiment_id, 'experiment_summary.json')
        
        try:
            data = load_experiment_summary(summary_path)
            
            for domain in data:
                domain_dir = os.path.join('plots', f'{domain}_{experiment_id}')
                os.makedirs(domain_dir, exist_ok=True)
                
                for model in data[domain]:
                    # Create recall plot
                    recall_plot_path = os.path.join(domain_dir, f'{domain}_{model}_recall.png')
                    create_bar_plot(data, domain, model, 'recall', recall_plot_path)
                    
                    # Create correctness plot
                    correctness_plot_path = os.path.join(domain_dir, f'{domain}_{model}_correctness.png')
                    create_bar_plot(data, domain, model, 'correct', correctness_plot_path)
                    
                    # Create summary table
                    table_path = os.path.join(domain_dir, f'{domain}_{model}_table.png')
                    create_summary_table(data, domain, model, table_path)
            
            logging.info(f"Completed processing experiment: {experiment_id}")
        except Exception as e:
            logging.error(f"Error processing experiment {experiment_id}: {str(e)}")
            continue

if __name__ == '__main__':
    main() 