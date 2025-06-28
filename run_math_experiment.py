#!/usr/bin/env python3
"""
Simple script to run math experiments using the new framework
"""

from benchmark.math import run_math_experiment
from plot import plot

def main():
    """Run a simple math experiment"""
    
    # Run experiment with default settings
    results = run_math_experiment(
        models=["gpt-4o-mini","o1-mini", "o3-mini", "o4-mini"],
        max_samples=50,
        output_dir="results",
        with_cot=True,
        custom_id="mix_token_step_budget"
    )
    
    print("Experiment completed!")
    print("Results:", results)
    plot()

if __name__ == "__main__":
    main() 