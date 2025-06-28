"""
Experiment runner for math benchmark.
"""

import os
import json
import logging
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
from pathlib import Path
import re

from benchmark.math.prompt import MathCallBuilder
from shared.evaluator import MathEvaluator
from shared.run import Call
from shared.model import Problem, CallResp
from shared.llm_client import LLMClient
from shared.experiment import Experiment, CallBuilder
from .dataset import MathDataset

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)




formalize_prompt = """First, formalize the problem by defining all variables, constraints, and the goal. Then, solve using only the formal structure."""

def run_math_experiment(
    domains: List[str] = None,
    models: List[str] = None,
    max_samples: int = None,
    with_cot: bool = True,
    output_dir: str = "results/math",
    custom_id: str = None
) -> Dict[str, Any]:
    """Run math experiment
    
    Args:
        domains: List of domains to test. Defaults to ["all"]
        models: List of models to test. Defaults to ["gpt-3.5-turbo"]
        max_samples: Maximum number of samples per domain
        with_cot: Whether to use chain-of-thought prompting
        output_dir: Output directory for results
        custom_id: Optional custom experiment ID. If not provided, an ID will be auto-generated
    """
    if domains is None:
        domains = ["all"]
    
    if models is None:
        models = ["gpt-3.5-turbo"]
    
    dataset = MathDataset()
    call_builder = MathCallBuilder()
    
    experiment = Experiment(
        dataset=dataset,
        call_builder=call_builder,
        output_dir=output_dir,
        with_cot=with_cot,
        evaluator=MathEvaluator(),
        custom_id=custom_id
    )
    
    return experiment.run(domains=domains, models=models, max_samples=max_samples)

def main():
    """Command-line interface."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Math CEP Experiment")
    parser.add_argument("--models", type=str, nargs="+", 
                       default=["gpt-3.5-turbo"],
                       help="Models to test")
    parser.add_argument("--max_samples", type=int, default=50,
                       help="Maximum number of samples to test")
    parser.add_argument("--output_dir", type=str, default="math_results",
                       help="Output directory for results")
    parser.add_argument("--data_dir", type=str, default="./data",
                       help="Directory containing dataset files")
    parser.add_argument("--all_models", action="store_true",
                       help="Use all available models")
    parser.add_argument("--experiment_id", type=str,
                       help="Custom experiment ID. If not provided, an ID will be auto-generated")
    
    args = parser.parse_args()
    
    # Check for OpenAI API key
    if not os.getenv("OPENAI_API_KEY"):
        logger.error("OPENAI_API_KEY not found in environment variables")
        logger.error("Please set your OpenAI API key:")
        logger.error("export OPENAI_API_KEY='your-api-key-here'")
        return
    
    # Handle --all_models flag
    if args.all_models:
        models = ["gpt-3.5-turbo", "gpt-4", "gpt-4-turbo"]
        logger.info(f"Using all models: {models}")
    else:
        models = args.models
    
    results = run_math_experiment(
        models=models,
        max_samples=args.max_samples,
        output_dir=args.output_dir,
        custom_id=args.experiment_id
    )
    
    logger.info("Experiment completed!")
    logger.info(f"Results saved to: {args.output_dir}")

if __name__ == "__main__":
    main() 