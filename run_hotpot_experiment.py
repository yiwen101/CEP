#!/usr/bin/env python3
"""
Simple script to run HotPotQA experiments using the new framework
"""

from benchmark.hotpot import run_hotpot_experiment

def main():
    """Run a simple HotPotQA experiment"""
    
    # Run experiment with default settings
    results = run_hotpot_experiment(
        domains=["hotpot_dev_distractor"],
        models=["gpt-3.5-turbo"],
        max_samples=2,
    )
    
    print("Experiment completed!")
    print("Results:", results)

if __name__ == "__main__":
    main() 