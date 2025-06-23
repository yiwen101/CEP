#!/usr/bin/env python3
"""
MuSR CEP Experiment using the new framework
"""

import os
import argparse
from typing import List
from dotenv import load_dotenv

from shared import Experiment, Evaluator
from .dataset import MusrDataset
from .calls import MusrCallBuilder

# Load environment variables
load_dotenv()

def run_musr_experiment(
    domains: List[str],
    models: List[str], 
    max_samples: int = 50,
    output_dir: str = "results",
    with_cot: bool = True
) -> dict:
    """
    Run MuSR CEP experiment using the new framework
    
    Args:
        domains: List of MuSR domains to test
        models: List of models to test
        max_samples: Maximum number of samples per domain
        output_dir: Output directory for results
        
    Returns:
        Dictionary with experiment results
    """
    print(f"Starting MuSR CEP Experiment")
    print(f"Domains: {domains}")
    print(f"Models: {models}")
    print(f"Max samples per domain: {max_samples}")
    print(f"Output directory: {output_dir}")
    
    # Create dataset and call builder
    dataset = MusrDataset()
    call_builder = MusrCallBuilder()
    
    # Create experiment
    experiment = Experiment(
        dataset=dataset,
        call_builder=call_builder,
        output_dir=output_dir,
        with_cot=with_cot
    )

    # Run experiment
    results = experiment.run(
        domains=domains,
        models=models,
        max_samples=max_samples
    )
    
    
    return results

def main():
    """Main function for command-line interface"""
    parser = argparse.ArgumentParser(description="MuSR CEP Experiment")
    parser.add_argument("--domains", type=str, nargs="+", 
                       default=["team_allocation"],
                       help="MuSR domains to test (murder_mystery, object_placements, team_allocation)")
    parser.add_argument("--models", type=str, nargs="+", 
                       default=["gpt-3.5-turbo"],
                       help="OpenAI models to test")
    parser.add_argument("--max_samples", type=int, default=50,
                       help="Maximum number of samples per domain")
    parser.add_argument("--output_dir", type=str, default="musr_results",
                       help="Output directory for results")
    parser.add_argument("--all_models", action="store_true",
                       help="Use all available models (gpt-3.5-turbo, gpt-4o, gpt-4o-mini)")
    parser.add_argument("--all_domains", action="store_true",
                       help="Use all available MuSR domains")
    
    args = parser.parse_args()
    
    # Check for OpenAI API key
    if not os.getenv("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY not found in environment variables.")
        print("Please set your OpenAI API key in a .env file:")
        print("echo 'OPENAI_API_KEY=your-api-key-here' > .env")
        print("Or set it as an environment variable:")
        print("export OPENAI_API_KEY='your-api-key-here'")
        return
    
    # Handle --all_models flag
    if args.all_models:
        models = ["gpt-3.5-turbo", "gpt-4o", "gpt-4o-mini"]
        print(f"Using all models: {models}")
    else:
        models = args.models
    
    # Handle --all_domains flag
    if args.all_domains:
        domains = ["murder_mystery", "object_placements", "team_allocation"]
        print(f"Using all domains: {domains}")
    else:
        domains = args.domains
    
    # Validate domains
    available_domains = ["murder_mystery", "object_placements", "team_allocation"]
    for domain in domains:
        if domain not in available_domains:
            print(f"Warning: Unknown domain '{domain}'. Available domains: {available_domains}")
    
    # Run experiment
    try:
        results = run_musr_experiment(
            domains=domains,
            models=models,
            max_samples=args.max_samples,
            output_dir=args.output_dir
        )
        
        print(f"\nMuSR CEP Experiment completed successfully!")
        print(f"Results saved to: {args.output_dir}")
        
    except Exception as e:
        print(f"Error running experiment: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 