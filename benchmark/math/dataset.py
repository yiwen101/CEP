"""
Dataset handling for math benchmark.
"""

import os
import json
import logging
from typing import List, Dict, Any, Optional
from shared.model import Problem
from shared.evaluator import MathEvaluator
from shared.experiment import Dataset

logger = logging.getLogger(__name__)

class MathDataset(Dataset):
    """Handler for math dataset."""
    
    def __init__(self, data_dir: str = "./data"):
        """
        Initialize math dataset handler.
        
        Args:
            data_dir: Base directory containing dataset files
        """
        self.data_dir = data_dir
        
    def get_dataset_name(self) -> str:
        """Return the name of this dataset"""
        return "math"
        
    def get_domains(self) -> List[str]:
        """Return list of available domains in this dataset"""
        return ["all"]  # Math dataset has a single domain
        
    def get_problems(self, domain: str, max_samples: int) -> List[Problem]:
        """
        Return list of problems for a given domain.
        
        Args:
            domain: Domain to get problems for (only "all" is supported)
            max_samples: Maximum number of samples to return
            
        Returns:
            List of Problem objects
        """
        if domain != "all":
            raise ValueError(f"Invalid domain '{domain}'. Only 'all' is supported.")
            
        data_file = os.path.join(self.data_dir, "math", "test.jsonl")
        
        if not os.path.exists(data_file):
            raise FileNotFoundError(f"Dataset file not found: {data_file}")
            
        problems = []
        with open(data_file, 'r') as f:
            for i, line in enumerate(f):
                if max_samples and i >= max_samples:
                    break
                    
                data = json.loads(line)
                problem = Problem(
                    question=data.get('problem', ''),
                    context='',  # Math problems typically don't have context
                    gold_answer=str(data.get('answer', '')),
                    problem_id=str(data.get('idx', i)),
                    choices=[]  # Math problems typically don't have choices
                )
                problems.append(problem)
                
        logger.info(f"Loaded {len(problems)} problems from {data_file}")
        return problems
    
    def format_prompt(self, problem: Problem) -> str:
        """
        Format a problem into a prompt for the model.
        
        Args:
            problem: Problem object
            
        Returns:
            Formatted prompt string
        """
        # Format the prompt with system instruction
        prompt = (
            "Please solve this math problem step by step. "
            "Put your final answer within \\boxed{}.\n\n"
            f"Question: {problem.question}\n\n"
            "Solution:"
        )
        
        return prompt