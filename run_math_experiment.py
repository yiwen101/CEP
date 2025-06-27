#!/usr/bin/env python3
"""
Simple script to run math experiments using the new framework
"""

from benchmark.math import run_math_experiment

def main():
    """Run a simple math experiment"""
    
    # Run experiment with default settings
    results = run_math_experiment(
        models=["gpt-4o-mini"],
        max_samples=500,
        output_dir="results",
        with_cot=True
    )
    
    print("Experiment completed!")
    print("Results:", results)

if __name__ == "__main__":
    main() 