#!/usr/bin/env python3
"""
Analysis script for CEP experiment results
Provides detailed analysis and visualization of experiment outcomes
"""

import json
import os
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List
import pandas as pd

def load_results(results_dir: str) -> Dict:
    """Load all results from the results directory"""
    results = {}
    
    # Load summary
    summary_file = os.path.join(results_dir, "experiment_summary.json")
    if os.path.exists(summary_file):
        with open(summary_file, 'r') as f:
            results['summary'] = json.load(f)
    
    # Load detailed results
    for filename in os.listdir(results_dir):
        if filename.endswith('_results.json') and filename != 'experiment_summary.json':
            method_name = filename.replace('_results.json', '')
            filepath = os.path.join(results_dir, filename)
            with open(filepath, 'r') as f:
                results[method_name] = json.load(f)
    
    return results

def create_performance_comparison(summary: Dict) -> None:
    """Create performance comparison charts"""
    methods = list(summary.keys())
    em_scores = [summary[method]['exact_match'] for method in methods]
    f1_scores = [summary[method]['f1_score'] for method in methods]
    llm_scores = [summary[method].get('llm_correctness', 0) for method in methods]
    execution_times = [summary[method]['avg_execution_time'] for method in methods]
    token_usage = [summary[method]['avg_tokens_used'] for method in methods]
    
    # Create subplots
    fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2, 3, figsize=(18, 12))
    
    # Exact Match scores
    bars1 = ax1.bar(methods, em_scores, color='skyblue', alpha=0.7)
    ax1.set_title('Exact Match Scores')
    ax1.set_ylabel('Exact Match')
    ax1.tick_params(axis='x', rotation=45)
    ax1.set_ylim(0, max(em_scores) * 1.1)
    
    # Add value labels on bars
    for bar, score in zip(bars1, em_scores):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{score:.3f}', ha='center', va='bottom')
    
    # F1 scores
    bars2 = ax2.bar(methods, f1_scores, color='lightgreen', alpha=0.7)
    ax2.set_title('F1 Scores')
    ax2.set_ylabel('F1 Score')
    ax2.tick_params(axis='x', rotation=45)
    ax2.set_ylim(0, max(f1_scores) * 1.1)
    
    for bar, score in zip(bars2, f1_scores):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{score:.3f}', ha='center', va='bottom')
    
    # LLM Correctness scores
    bars3 = ax3.bar(methods, llm_scores, color='orange', alpha=0.7)
    ax3.set_title('LLM Correctness Scores')
    ax3.set_ylabel('LLM Correctness')
    ax3.tick_params(axis='x', rotation=45)
    ax3.set_ylim(0, max(llm_scores) * 1.1 if max(llm_scores) > 0 else 1.1)
    
    for bar, score in zip(bars3, llm_scores):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{score:.3f}', ha='center', va='bottom')
    
    # Execution times
    bars4 = ax4.bar(methods, execution_times, color='salmon', alpha=0.7)
    ax4.set_title('Average Execution Time')
    ax4.set_ylabel('Time (seconds)')
    ax4.tick_params(axis='x', rotation=45)
    
    for bar, time_val in zip(bars4, execution_times):
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                f'{time_val:.2f}s', ha='center', va='bottom')
    
    # Token usage
    bars5 = ax5.bar(methods, token_usage, color='gold', alpha=0.7)
    ax5.set_title('Average Token Usage')
    ax5.set_ylabel('Tokens')
    ax5.tick_params(axis='x', rotation=45)
    
    for bar, tokens in zip(bars5, token_usage):
        ax5.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 50,
                f'{tokens:.0f}', ha='center', va='bottom')
    
    # Combined accuracy comparison
    x = np.arange(len(methods))
    width = 0.25
    
    ax6.bar(x - width, em_scores, width, label='Exact Match', alpha=0.7)
    ax6.bar(x, f1_scores, width, label='F1 Score', alpha=0.7)
    ax6.bar(x + width, llm_scores, width, label='LLM Correctness', alpha=0.7)
    
    ax6.set_title('Accuracy Metrics Comparison')
    ax6.set_ylabel('Score')
    ax6.set_xticks(x)
    ax6.set_xticklabels(methods, rotation=45)
    ax6.legend()
    
    plt.tight_layout()
    plt.savefig('performance_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

def create_efficiency_analysis(summary: Dict) -> None:
    """Create efficiency analysis (performance vs. cost)"""
    methods = list(summary.keys())
    f1_scores = [summary[method]['f1_score'] for method in methods]
    llm_scores = [summary[method].get('llm_correctness', 0) for method in methods]
    token_usage = [summary[method]['avg_tokens_used'] for method in methods]
    execution_times = [summary[method]['avg_execution_time'] for method in methods]
    
    # Calculate efficiency metrics
    efficiency_f1_per_token = [f1/token for f1, token in zip(f1_scores, token_usage)]
    efficiency_llm_per_token = [llm/token for llm, token in zip(llm_scores, token_usage)]
    efficiency_f1_per_time = [f1/time for f1, time in zip(f1_scores, execution_times)]
    efficiency_llm_per_time = [llm/time for llm, time in zip(llm_scores, execution_times)]
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # F1 per token
    bars1 = ax1.bar(methods, efficiency_f1_per_token, color='purple', alpha=0.7)
    ax1.set_title('F1 Score per Token (Efficiency)')
    ax1.set_ylabel('F1 / Token')
    ax1.tick_params(axis='x', rotation=45)
    
    for bar, eff in zip(bars1, efficiency_f1_per_token):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.0001,
                f'{eff:.4f}', ha='center', va='bottom')
    
    # LLM per token
    bars2 = ax2.bar(methods, efficiency_llm_per_token, color='green', alpha=0.7)
    ax2.set_title('LLM Correctness per Token (Efficiency)')
    ax2.set_ylabel('LLM / Token')
    ax2.tick_params(axis='x', rotation=45)
    
    for bar, eff in zip(bars2, efficiency_llm_per_token):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.0001,
                f'{eff:.4f}', ha='center', va='bottom')
    
    # F1 per second
    bars3 = ax3.bar(methods, efficiency_f1_per_time, color='orange', alpha=0.7)
    ax3.set_title('F1 Score per Second (Speed)')
    ax3.set_ylabel('F1 / Second')
    ax3.tick_params(axis='x', rotation=45)
    
    for bar, eff in zip(bars3, efficiency_f1_per_time):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{eff:.3f}', ha='center', va='bottom')
    
    # LLM per second
    bars4 = ax4.bar(methods, efficiency_llm_per_time, color='red', alpha=0.7)
    ax4.set_title('LLM Correctness per Second (Speed)')
    ax4.set_ylabel('LLM / Second')
    ax4.tick_params(axis='x', rotation=45)
    
    for bar, eff in zip(bars4, efficiency_llm_per_time):
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{eff:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('efficiency_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

def create_metric_correlation_analysis(summary: Dict) -> None:
    """Create correlation analysis between different metrics"""
    methods = list(summary.keys())
    em_scores = [summary[method]['exact_match'] for method in methods]
    f1_scores = [summary[method]['f1_score'] for method in methods]
    llm_scores = [summary[method].get('llm_correctness', 0) for method in methods]
    token_usage = [summary[method]['avg_tokens_used'] for method in methods]
    execution_times = [summary[method]['avg_execution_time'] for method in methods]
    
    # Create correlation matrix
    data = {
        'EM': em_scores,
        'F1': f1_scores,
        'LLM': llm_scores,
        'Tokens': token_usage,
        'Time': execution_times
    }
    
    df = pd.DataFrame(data)
    correlation_matrix = df.corr()
    
    # Plot correlation heatmap
    plt.figure(figsize=(12, 10))
    plt.imshow(correlation_matrix, cmap='coolwarm', aspect='auto', vmin=-1, vmax=1)
    plt.colorbar(label='Correlation Coefficient')
    plt.xticks(range(len(correlation_matrix.columns)), correlation_matrix.columns)
    plt.yticks(range(len(correlation_matrix.columns)), correlation_matrix.columns)
    plt.title('Metric Correlation Matrix')
    
    # Add correlation values to the plot
    for i in range(len(correlation_matrix.columns)):
        for j in range(len(correlation_matrix.columns)):
            plt.text(j, i, f'{correlation_matrix.iloc[i, j]:.2f}', 
                    ha='center', va='center', fontsize=10)
    
    plt.tight_layout()
    plt.savefig('metric_correlation.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print correlation insights
    print("\n" + "="*60)
    print("METRIC CORRELATION INSIGHTS")
    print("="*60)
    
    # EM vs F1 correlation
    em_f1_corr = correlation_matrix.loc['EM', 'F1']
    print(f"EM vs F1 correlation: {em_f1_corr:.3f}")
    
    # EM vs LLM correlation
    em_llm_corr = correlation_matrix.loc['EM', 'LLM']
    print(f"EM vs LLM correlation: {em_llm_corr:.3f}")
    
    # F1 vs LLM correlation
    f1_llm_corr = correlation_matrix.loc['F1', 'LLM']
    print(f"F1 vs LLM correlation: {f1_llm_corr:.3f}")
    
    # Performance vs cost correlations
    em_tokens_corr = correlation_matrix.loc['EM', 'Tokens']
    print(f"EM vs Token usage correlation: {em_tokens_corr:.3f}")
    
    f1_tokens_corr = correlation_matrix.loc['F1', 'Tokens']
    print(f"F1 vs Token usage correlation: {f1_tokens_corr:.3f}")

def analyze_reasoning_patterns(results: Dict) -> None:
    """Analyze patterns in LLM reasoning for evaluation"""
    print("\n" + "="*80)
    print("LLM REASONING PATTERN ANALYSIS")
    print("="*80)
    
    for method_name, method_results in results.items():
        if method_name == 'summary':
            continue
            
        print(f"\n{method_name}:")
        
        # Count LLM correctness
        correct_count = 0
        total_count = 0
        
        for result in method_results:
            if result.get('llm_correctness') is not None:
                total_count += 1
                if result['llm_correctness']:
                    correct_count += 1
        
        if total_count > 0:
            accuracy = correct_count / total_count
            print(f"  LLM Evaluation Accuracy: {accuracy:.3f} ({correct_count}/{total_count})")

def analyze_method_categories(summary: Dict) -> None:
    """Analyze performance by method categories"""
    categories = {
        'Baseline': [k for k in summary.keys() if k.startswith('baseline')],
        'ICE with Intention': [k for k in summary.keys() if 'ice_with_intention' in k],
        'ICE without Intention': [k for k in summary.keys() if 'ice_without_intention' in k],
        'ICE Context Augmentation': [k for k in summary.keys() if 'ice_context_augmentation' in k]
    }
    
    print("\n" + "="*80)
    print("PERFORMANCE BY METHOD CATEGORIES")
    print("="*80)
    
    for category, methods in categories.items():
        if methods:
            em_scores = [summary[method]['exact_match'] for method in methods]
            f1_scores = [summary[method]['f1_score'] for method in methods]
            llm_scores = [summary[method].get('llm_correctness', 0) for method in methods]
            
            print(f"\n{category}:")
            print(f"  Methods: {', '.join(methods)}")
            print(f"  Average EM: {np.mean(em_scores):.3f} ± {np.std(em_scores):.3f}")
            print(f"  Average F1: {np.mean(f1_scores):.3f} ± {np.std(f1_scores):.3f}")
            print(f"  Average LLM: {np.mean(llm_scores):.3f} ± {np.std(llm_scores):.3f}")
            print(f"  Best EM: {max(em_scores):.3f} ({methods[np.argmax(em_scores)]})")
            print(f"  Best F1: {max(f1_scores):.3f} ({methods[np.argmax(f1_scores)]})")
            print(f"  Best LLM: {max(llm_scores):.3f} ({methods[np.argmax(llm_scores)]})")

def analyze_cep_types(summary: Dict) -> None:
    """Analyze performance by CEP type"""
    cep_types = ['understand', 'connect', 'query', 'application', 'comprehensive']
    
    print("\n" + "="*80)
    print("PERFORMANCE BY CEP TYPE")
    print("="*80)
    
    for cep_type in cep_types:
        methods = [k for k in summary.keys() if cep_type in k]
        if methods:
            em_scores = [summary[method]['exact_match'] for method in methods]
            f1_scores = [summary[method]['f1_score'] for method in methods]
            llm_scores = [summary[method].get('llm_correctness', 0) for method in methods]
            
            print(f"\n{cep_type.upper()}:")
            print(f"  Methods: {', '.join(methods)}")
            print(f"  Average EM: {np.mean(em_scores):.3f} ± {np.std(em_scores):.3f}")
            print(f"  Average F1: {np.mean(f1_scores):.3f} ± {np.std(f1_scores):.3f}")
            print(f"  Average LLM: {np.mean(llm_scores):.3f} ± {np.std(llm_scores):.3f}")

def generate_report(results_dir: str, output_file: str = "analysis_report.txt") -> None:
    """Generate a comprehensive analysis report"""
    results = load_results(results_dir)
    
    if 'summary' not in results:
        print(f"No summary found in {results_dir}")
        return
    
    summary = results['summary']
    
    with open(output_file, 'w') as f:
        f.write("CEP EXPERIMENT ANALYSIS REPORT\n")
        f.write("=" * 50 + "\n\n")
        
        # Overall summary
        f.write("OVERALL SUMMARY\n")
        f.write("-" * 20 + "\n")
        f.write(f"Total methods tested: {len(summary)}\n")
        f.write(f"Best EM: {max([summary[m]['exact_match'] for m in summary]):.3f}\n")
        f.write(f"Best F1: {max([summary[m]['f1_score'] for m in summary]):.3f}\n")
        f.write(f"Best LLM: {max([summary[m].get('llm_correctness', 0) for m in summary]):.3f}\n\n")
        
        # Method rankings
        f.write("METHOD RANKINGS\n")
        f.write("-" * 20 + "\n")
        
        # Sort by EM
        sorted_by_em = sorted(summary.items(), key=lambda x: x[1]['exact_match'], reverse=True)
        f.write("Ranked by Exact Match:\n")
        for i, (method, metrics) in enumerate(sorted_by_em, 1):
            llm_score = metrics.get('llm_correctness', 0)
            f.write(f"{i}. {method}: EM={metrics['exact_match']:.3f}, F1={metrics['f1_score']:.3f}, LLM={llm_score:.3f}\n")
        
        f.write("\nRanked by F1 Score:\n")
        sorted_by_f1 = sorted(summary.items(), key=lambda x: x[1]['f1_score'], reverse=True)
        for i, (method, metrics) in enumerate(sorted_by_f1, 1):
            llm_score = metrics.get('llm_correctness', 0)
            f.write(f"{i}. {method}: F1={metrics['f1_score']:.3f}, EM={metrics['exact_match']:.3f}, LLM={llm_score:.3f}\n")
        
        f.write("\nRanked by LLM Correctness:\n")
        sorted_by_llm = sorted(summary.items(), key=lambda x: x[1].get('llm_correctness', 0), reverse=True)
        for i, (method, metrics) in enumerate(sorted_by_llm, 1):
            llm_score = metrics.get('llm_correctness', 0)
            f.write(f"{i}. {method}: LLM={llm_score:.3f}, EM={metrics['exact_match']:.3f}, F1={metrics['f1_score']:.3f}\n")
        
        # Efficiency analysis
        f.write("\nEFFICIENCY ANALYSIS\n")
        f.write("-" * 20 + "\n")
        for method, metrics in summary.items():
            f1_per_token = metrics['f1_score'] / metrics['avg_tokens_used']
            f1_per_second = metrics['f1_score'] / metrics['avg_execution_time']
            llm_per_token = metrics.get('llm_correctness', 0) / metrics['avg_tokens_used']
            llm_per_second = metrics.get('llm_correctness', 0) / metrics['avg_execution_time']
            
            f.write(f"{method}:\n")
            f.write(f"  F1 per token: {f1_per_token:.6f}\n")
            f.write(f"  F1 per second: {f1_per_second:.3f}\n")
            f.write(f"  LLM per token: {llm_per_token:.6f}\n")
            f.write(f"  LLM per second: {llm_per_second:.3f}\n\n")
    
    print(f"Analysis report saved to {output_file}")

def main():
    """Main analysis function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Analyze CEP experiment results")
    parser.add_argument("--results_dir", type=str, default="results",
                       help="Directory containing experiment results")
    parser.add_argument("--output_file", type=str, default="analysis_report.txt",
                       help="Output file for analysis report")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.results_dir):
        print(f"Results directory {args.results_dir} not found!")
        return
    
    results = load_results(args.results_dir)
    
    if 'summary' not in results:
        print(f"No summary found in {args.results_dir}")
        return
    
    summary = results['summary']
    
    print("CEP EXPERIMENT ANALYSIS")
    print("=" * 30)
    
    # Generate visualizations
    try:
        create_performance_comparison(summary)
        create_efficiency_analysis(summary)
        create_metric_correlation_analysis(summary)
        print("Charts saved as:")
        print("- performance_comparison.png")
        print("- efficiency_analysis.png") 
        print("- metric_correlation.png")
    except ImportError:
        print("matplotlib/pandas not available, skipping visualizations")
    
    # Generate text analysis
    analyze_method_categories(summary)
    analyze_cep_types(summary)
    analyze_reasoning_patterns(results)
    
    # Generate report
    generate_report(args.results_dir, args.output_file)
    
    print(f"\nAnalysis complete! Check {args.output_file} for detailed report.")

if __name__ == "__main__":
    main() 