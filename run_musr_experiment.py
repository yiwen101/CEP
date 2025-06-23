#!/usr/bin/env python3
"""
Simple script to run MuSR experiments using the new framework
"""

from benchmark.musr import run_musr_experiment

def main():
    """Run a simple MuSR experiment"""
    
    # Run experiment with default settings
    results = run_musr_experiment(
        domains=["murder_mystery"],
        models=["gpt-3.5-turbo", "gpt-4o"],
        max_samples=50
    )
    
    print("Experiment completed!")
    print("Results:", results)

if __name__ == "__main__":
    main() 