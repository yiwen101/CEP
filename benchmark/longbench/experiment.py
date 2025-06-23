#!/usr/bin/env python3
"""
LongBenchV2 CEP Experiment using the new framework
"""

import os
import argparse
from typing import List, Optional
from dotenv import load_dotenv

from shared import Experiment, Evaluator
from .dataset import LongBenchDataset
from .calls import LongBenchCallBuilder

# Load environment variables
load_dotenv()

def run_longbench_experiment(
    domains: Optional[List[str]] = None,
    models: List[str] = ["gpt-4o-mini-2024-07-18"], 
    max_samples: int = 50,
    output_dir: str = "results",
    with_cot: bool = False
) -> dict:
    """
    Run LongBenchV2 CEP experiment using the new framework
    
    Args:
        domains: List of LongBenchV2 domains to test (if None, uses all available)
        models: List of models to test
        max_samples: Maximum number of samples per domain
        split: Dataset split ('train' or 'test')
        output_dir: Output directory for results
        with_cot: Whether to use Chain-of-Thought prompting
        
    Returns:
        Dictionary with experiment results
    """
    print(f"Starting LongBenchV2 CEP Experiment")
    print(f"Models: {models}")
    print(f"Max samples: {max_samples}")
    print(f"Output directory: {output_dir}")
    print(f"Chain-of-Thought: {with_cot}")
    
    # Create dataset and call builder
    dataset = LongBenchDataset( max_samples=max_samples)
    print(dataset.get_domains())
    call_builder = LongBenchCallBuilder()
    
    # Get available domains if not specified
    if domains is None:
        domains = dataset.get_domains()
        print(f"Using all available domains: {domains}")
    else:
        print(f"Domains: {domains}")
    
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
    parser = argparse.ArgumentParser(description="LongBenchV2 CEP Experiment")
    parser.add_argument("--domains", type=str, nargs="+", 
                       help="LongBenchV2 domains to test (if not specified, uses all available)")
    parser.add_argument("--models", type=str, nargs="+", 
                       default=["gpt-4o-mini-2024-07-18"],
                       help="OpenAI models to test")
    parser.add_argument("--max_samples", type=int, default=50,
                       help="Maximum number of samples")
    parser.add_argument("--split", type=str, default="train",
                       choices=["train", "test"],
                       help="Dataset split to use")
    parser.add_argument("--output_dir", type=str, default="results",
                       help="Output directory for results")
    parser.add_argument("--all_models", action="store_true",
                       help="Use all available models (gpt-3.5-turbo, gpt-4o, gpt-4o-mini)")
    parser.add_argument("--with_cot", action="store_true",
                       help="Use Chain-of-Thought prompting")
    
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
        models = ["gpt-3.5-turbo", "gpt-4o", "gpt-4o-mini-2024-07-18"]
        print(f"Using all models: {models}")
    else:
        models = args.models
    
    # Validate models (basic check)
    available_models = ["gpt-3.5-turbo", "gpt-4o", "gpt-4o-mini-2024-07-18", "gpt-4o-mini"]
    for model in models:
        if model not in available_models:
            print(f"Warning: Model '{model}' not in known list. Available models: {available_models}")
    
    # Run experiment
    try:
        results = run_longbench_experiment(
            domains=args.domains,
            models=models,
            max_samples=args.max_samples,
            split=args.split,
            output_dir=args.output_dir,
            with_cot=args.with_cot
        )
        
        print(f"\nLongBenchV2 CEP Experiment completed successfully!")
        print(f"Results saved to: {args.output_dir}")
        
    except Exception as e:
        print(f"Error running experiment: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 