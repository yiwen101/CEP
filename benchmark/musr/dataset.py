"""
MuSR Dataset Implementation
"""

import json
import os
from typing import List, Dict, Any
from pathlib import Path

from shared import Dataset, Problem

class MusrDataset(Dataset):
    """Dataset implementation for MuSR (Multi-step Soft Reasoning)"""
    
    def __init__(self, data_dir: str = "data/musr"):
        """
        Initialize MuSR dataset
        
        Args:
            data_dir: Directory containing MuSR dataset files
        """
        self.data_dir = Path(data_dir)
        
        # Domain to filename mapping
        self.domain_files = {
            "murder_mystery": "murder_mystery.json",
            "object_placements": "object_placements.json", 
            "team_allocation": "team_allocation.json"
        }
        
        # Check if data directory exists
        if not self.data_dir.exists():
            raise FileNotFoundError(f"MuSR data directory not found: {self.data_dir}")
        
        # Check if all domain files exist
        for domain, filename in self.domain_files.items():
            file_path = self.data_dir / filename
            if not file_path.exists():
                print(f"Warning: MuSR domain file not found: {file_path}")
    
    def get_dataset_name(self) -> str:
        """Return the name of this dataset"""
        return "musr"
    
    def get_domains(self) -> List[str]:
        """Return list of available domains in this dataset"""
        available_domains = []
        for domain, filename in self.domain_files.items():
            file_path = self.data_dir / filename
            if file_path.exists():
                available_domains.append(domain)
        return available_domains
    
    def get_problems(self, domain: str, max_samples: int) -> List[Problem]:
        """Return list of problems for a given domain"""
        if domain not in self.domain_files:
            raise ValueError(f"Unknown domain: {domain}. Available domains: {list(self.domain_files.keys())}")
        
        filename = self.domain_files[domain]
        file_path = self.data_dir / filename
        
        if not file_path.exists():
            raise FileNotFoundError(f"MuSR domain file not found: {file_path}")
        
        print(f"Loading MuSR {domain} data from {file_path}")
        
        # Load data from JSON file
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        problems = []
        sample_count = 0
        
        for sample in data:
            if sample_count >= max_samples:
                break
                
            context = sample["context"]
            
            # Each sample can have multiple questions
            for question_idx, question_data in enumerate(sample["questions"]):
                if sample_count >= max_samples:
                    break
                
                # Create problem ID
                problem_id = f"{domain}_{sample_count}_{question_idx}"
                
                # Get the correct answer (MuSR uses 1-indexed answers)
                correct_choice_idx = question_data["answer"] - 1
                gold_answer = question_data["choices"][correct_choice_idx]
                
                # Create problem
                problem = Problem(
                    question=f"""{question_data["question"]}\n\nPick one of the following choices:\n{question_data["choices"]}\n\nYou must pick one option. Finally, the last thing you generate should be "ANSWER: (your answer here, include the choice number and content of the choice)""",
                    context=context,
                    gold_answer=gold_answer,
                    problem_id=problem_id
                )
                
                problems.append(problem)
                sample_count += 1
        
        print(f"Loaded {len(problems)} problems from {domain}")
        return problems 