#!/usr/bin/env python3
"""
Simple script to run LongBenchV2 experiments using the new framework
"""

from benchmark.longbench import run_longbench_experiment

def main():
    """Run a simple LongBenchV2 experiment"""
    
    # Run experiment with default settings
    results = run_longbench_experiment(
        domains=['Single-Document QA_short_easy', 'Single-Document QA_short_hard'],
        models=["gpt-4o-mini"],
        max_samples=50
    )
    
    print("Experiment completed!")
    print("Results:", results)

if __name__ == "__main__":
    main() 