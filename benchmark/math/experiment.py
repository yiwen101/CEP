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

from shared.evaluator import MathEvaluator
from shared.run import Call
from shared.model import Problem, CallResp
from shared.llm_client import LLMClient
from shared.experiment import Experiment, CallBuilder
from .dataset import MathDataset

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def extract_answer(pred_str, use_last_number=True):
    pred_str = pred_str.replace("\u043a\u0438", "")
    
    pred = ""
    
    if "boxed" in pred_str:
        ans = pred_str.split("boxed")[-1]
        if len(ans) == 0:
            return ""
        elif ans[0] == "{":
            stack = 1
            a = ""
            for c in ans[1:]:
                if c == "{":
                    stack += 1
                    a += c
                elif c == "}":
                    stack -= 1
                    if stack == 0:
                        break
                    a += c
                else:
                    a += c
        else:
            a = ans.split("$")[0].strip()
        pred = a

    # multiple line
    # pred = pred.split("\n")[0]
    pred = re.sub(r"\n\s*", "", pred)
    if pred != "" and pred[0] == ":":
        pred = pred[1:]
    if pred != "" and pred[-1] == ".":
        pred = pred[:-1]
    if pred != "" and pred[-1] == "/":
        pred = pred[:-1]
    # pred = strip_string(pred)
    return pred

def create_math_call(model: str) -> Call:
    """Create a math-specific call that handles boxed answer extraction"""
    llm_client = LLMClient(model)
    
    def math_call(problem: Problem) -> CallResp:
        """Math call with built-in answer extraction"""
        system_msg = """Please solve this math problem step by step. Put your final answer within \boxed{}.

Question:

Solution:"""
        
        messages = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": problem.question}
        ]
        
        full_response, tokens_used, execution_time = llm_client.call_with_history(messages)
        
        # Extract answer from the boxed environment
        predicted_answer = extract_answer(full_response)
        
        # Create full chat history including the response
        full_history = messages + [{"role": "assistant", "content": full_response}]
        
        return CallResp(
            predicted_answer=predicted_answer,
            execution_time=execution_time,
            tokens_used=tokens_used,
            chat_history=full_history
        )
    
    return math_call

class MathCallBuilder(CallBuilder):
    """Call builder for math problems"""
    
    def build_calls(self, model: str, domain: str, with_cot: bool) -> Dict[str, Call]:
        """Build calls for math problems"""
        return {"math": create_math_call(model)}

def run_math_experiment(
    domains: List[str] = None,
    models: List[str] = None,
    max_samples: int = None,
    with_cot: bool = True,
    output_dir: str = "results/math"
) -> Dict[str, Any]:
    """Run math experiment"""
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
        evaluator = MathEvaluator()
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
        data_dir=args.data_dir
    )
    
    logger.info("Experiment completed!")
    logger.info(f"Results saved to: {args.output_dir}")

if __name__ == "__main__":
    main() 