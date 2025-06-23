"""
LongBenchV2 dataset interface for CEP experiments.
"""

from typing import List, Dict, Any, Optional
from datasets import load_dataset
import json
import os

from shared.model import Problem


class LongBenchDataset:
    """LongBenchV2 dataset interface for CEP experiments."""
    
    def __init__(self, split: str = "train", max_samples: Optional[int] = None):
        """
        Initialize LongBenchV2 dataset.
        
        Args:
            split: Dataset split ('train' or 'test')
            max_samples: Maximum number of samples to load (for testing)
        """
        self.split = split
        self.max_samples = max_samples
        self.data = {}
        self.load_dataset()
        
    def get_dataset_name(self) -> str:
        """Return the name of this dataset"""
        return "longbench"
        
    def load_dataset(self):
        """Load LongBenchV2 dataset from HuggingFace."""
        try:
            dataset = None
            data_path = "data/longbench/data.json"
            if os.path.exists(data_path):
                with open(data_path, 'r', encoding='utf-8') as f:
                    dataset = json.load(f)
            else:
                dataset = load_dataset('THUDM/LongBench-v2', split=self.split)
                objs = []
                for item in dataset:
                    objs.append({
                        "_id": item["_id"],
                        "domain": item["domain"],
                        "sub_domain": item["sub_domain"],
                        "difficulty": item["difficulty"],
                        "length": item["length"],
                        "question": item["question"],
                        "choice_A": item["choice_A"],
                        "choice_B": item["choice_B"],
                        "choice_C": item["choice_C"],
                        "choice_D": item["choice_D"],
                        "answer": item["answer"],
                        "context": item["context"],
                    })
                # Create directory if it doesn't exist
                os.makedirs(os.path.dirname(data_path), exist_ok=True)
                with open(data_path, 'w', encoding='utf-8') as f:
                    json.dump(objs, f, ensure_ascii=False, indent=4)
            
            for item in dataset:
                choices = [
                    "A: " + item["choice_A"].strip(),
                    "B: " + item["choice_B"].strip(), 
                    "C: " + item["choice_C"].strip(),
                    "D: " + item["choice_D"].strip()
                ]
                
                # Create problem object
                problem = Problem(
                    problem_id=item["_id"],
                    question=f"What is the correct answer to this question: {item['question']}\nChoices:\n(A) {item['choice_A']}\n(B) {item['choice_B']}\n(C) {item['choice_C']}\n(D) {item['choice_D']}? Format your response as follows: \"The correct answer is (insert answer here)\"",
                    context=item["context"],
                    gold_answer=item["answer"],
                    choices=choices,
                )

                domain = f"{item['domain']}_{item['length']}_{item['difficulty']}"
                if domain not in self.data:
                    self.data[domain] = []
                
                self.data[domain].append(problem)
                
            
        except Exception as e:
            print(f"Error loading LongBenchV2 dataset: {e}")
            raise e
    
    def get_problems(self, domain: str, max_samples: int) -> List[Problem]:
        """Get problems for a given domain with max_samples limit."""
        if domain not in self.data:
            raise ValueError(f"Unknown domain: {domain}. Available domains: {list(self.data.keys())}")
        
        problems = self.data[domain]
        return problems[:max_samples]
    
    def get_domains(self) -> List[str]:
        """Get list of available domains."""
        return list(self.data.keys())
    
    def get_problems_by_domain(self, domain: str) -> List[Problem]:
        """Get problems filtered by domain."""
        if domain not in self.data:
            return []
        return self.data[domain]
    
    def get_problems_by_difficulty(self, difficulty: str) -> List[Problem]:
        """Get problems filtered by difficulty (easy/hard)."""
        problems = []
        for domain_problems in self.data.values():
            for problem in domain_problems:
                # Extract difficulty from domain name (format: domain_length_difficulty)
                domain_parts = problem.problem_id.split("_")
                if len(domain_parts) >= 3 and domain_parts[-1] == difficulty:
                    problems.append(problem)
        return problems
    
    def get_problems_by_length(self, length: str) -> List[Problem]:
        """Get problems filtered by length (short/medium/long)."""
        problems = []
        for domain_problems in self.data.values():
            for problem in domain_problems:
                # Extract length from domain name (format: domain_length_difficulty)
                domain_parts = problem.problem_id.split("_")
                if len(domain_parts) >= 3 and domain_parts[-2] == length:
                    problems.append(problem)
        return problems
    
    def get_sub_domains(self) -> List[str]:
        """Get list of available sub-domains."""
        sub_domains = set()
        for domain in self.data.keys():
            # Extract sub-domain from domain name
            parts = domain.split("_")
            if len(parts) >= 1:
                sub_domains.add(parts[0])
        return list(sub_domains)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get dataset statistics."""
        total = sum(len(problems) for problems in self.data.values())
        domains = {domain: len(problems) for domain, problems in self.data.items()}
        difficulties = {"easy": 0, "hard": 0}
        lengths = {"short": 0, "medium": 0, "long": 0}
        
        for domain, problems in self.data.items():
            for problem in problems:
                # Extract difficulty and length from domain name
                domain_parts = domain.split("_")
                if len(domain_parts) >= 3:
                    difficulty = domain_parts[-1]
                    length = domain_parts[-2]
                    
                    if difficulty in difficulties:
                        difficulties[difficulty] += 1
                    if length in lengths:
                        lengths[length] += 1
        
        return {
            "total_problems": total,
            "domains": domains,
            "difficulties": difficulties,
            "lengths": lengths
        }
    