"""
LongBenchV2 evaluator for CEP experiments.
"""

from typing import List, Dict, Any
from shared.model import Problem
from shared.evaluator import Evaluator


class LongBenchEvaluator(Evaluator):
    """LongBenchV2 evaluator for multiple-choice questions."""
    
    def __init__(self):
        super().__init__()
    
    def evaluate_single(self, problem: Problem, predicted_answer: str) -> Dict[str, Any]:
        """
        Evaluate a single LongBenchV2 problem.
        
        Args:
            problem: The problem with ground truth answer
            predicted_answer: Model's predicted answer
            
        Returns:
            Dictionary with evaluation results
        """
        # Extract the correct answer (A, B, C, or D)
        correct_answer = problem.answer
        
        # Clean and extract predicted answer
        predicted_clean = self._extract_choice(predicted_answer)
        
        # Check if answer is correct
        is_correct = predicted_clean == correct_answer
        
        return {
            "correct": is_correct,
            "predicted": predicted_clean,
            "ground_truth": correct_answer,
            "exact_match": is_correct
        }
    
    def evaluate_batch(self, problems: List[Problem], predicted_answers: List[str]) -> Dict[str, Any]:
        """
        Evaluate a batch of LongBenchV2 problems.
        
        Args:
            problems: List of problems
            predicted_answers: List of predicted answers
            
        Returns:
            Dictionary with aggregated evaluation results
        """
        if len(problems) != len(predicted_answers):
            raise ValueError("Number of problems and predicted answers must match")
        
        results = []
        for problem, pred in zip(problems, predicted_answers):
            result = self.evaluate_single(problem, pred)
            results.append(result)
        
        return self._aggregate_results(results, problems)
    
    def _extract_choice(self, answer: str) -> str:
        """
        Extract the choice (A, B, C, or D) from the model's response.
        
        Args:
            answer: Model's response
            
        Returns:
            Extracted choice (A, B, C, or D)
        """
        answer = answer.strip().upper()
        
        # Look for patterns like "The correct answer is (A)" or "Answer: A"
        import re
        
        # Pattern 1: "The correct answer is (A)"
        match = re.search(r'The correct answer is \(([A-D])\)', answer)
        if match:
            return match.group(1)
        
        # Pattern 2: "The correct answer is A"
        match = re.search(r'The correct answer is ([A-D])', answer)
        if match:
            return match.group(1)
        
        # Pattern 3: "Answer: A" or "A)" or "A."
        match = re.search(r'Answer:\s*([A-D])', answer)
        if match:
            return match.group(1)
        
        # Pattern 4: Just the letter A, B, C, or D
        match = re.search(r'\b([A-D])\b', answer)
        if match:
            return match.group(1)
        
        # If no clear answer found, return None
        return None
    
    def _aggregate_results(self, results: List[Dict[str, Any]], problems: List[Problem]) -> Dict[str, Any]:
        """
        Aggregate evaluation results with domain-specific breakdowns.
        
        Args:
            results: List of individual evaluation results
            problems: List of problems for metadata
            
        Returns:
            Aggregated results with breakdowns
        """
        total = len(results)
        correct = sum(1 for r in results if r["correct"])
        accuracy = correct / total if total > 0 else 0
        
        # Domain breakdown
        domain_stats = {}
        difficulty_stats = {"easy": {"total": 0, "correct": 0}, "hard": {"total": 0, "correct": 0}}
        length_stats = {"short": {"total": 0, "correct": 0}, "medium": {"total": 0, "correct": 0}, "long": {"total": 0, "correct": 0}}
        
        for result, problem in zip(results, problems):
            # Domain stats
            domain = problem.metadata.get("domain", "unknown")
            if domain not in domain_stats:
                domain_stats[domain] = {"total": 0, "correct": 0}
            domain_stats[domain]["total"] += 1
            if result["correct"]:
                domain_stats[domain]["correct"] += 1
            
            # Difficulty stats
            difficulty = problem.metadata.get("difficulty", "unknown")
            if difficulty in difficulty_stats:
                difficulty_stats[difficulty]["total"] += 1
                if result["correct"]:
                    difficulty_stats[difficulty]["correct"] += 1
            
            # Length stats
            length = problem.metadata.get("length", "unknown")
            if length in length_stats:
                length_stats[length]["total"] += 1
                if result["correct"]:
                    length_stats[length]["correct"] += 1
        
        # Calculate domain accuracies
        domain_accuracies = {}
        for domain, stats in domain_stats.items():
            domain_accuracies[domain] = stats["correct"] / stats["total"] if stats["total"] > 0 else 0
        
        # Calculate difficulty accuracies
        difficulty_accuracies = {}
        for difficulty, stats in difficulty_stats.items():
            difficulty_accuracies[difficulty] = stats["correct"] / stats["total"] if stats["total"] > 0 else 0
        
        # Calculate length accuracies
        length_accuracies = {}
        for length, stats in length_stats.items():
            length_accuracies[length] = stats["correct"] / stats["total"] if stats["total"] > 0 else 0
        
        return {
            "overall_accuracy": accuracy,
            "total_problems": total,
            "correct_answers": correct,
            "domain_breakdown": {
                "accuracies": domain_accuracies,
                "counts": {domain: stats["total"] for domain, stats in domain_stats.items()}
            },
            "difficulty_breakdown": {
                "accuracies": difficulty_accuracies,
                "counts": {diff: stats["total"] for diff, stats in difficulty_stats.items()}
            },
            "length_breakdown": {
                "accuracies": length_accuracies,
                "counts": {length: stats["total"] for length, stats in length_stats.items()}
            },
            "individual_results": results
        } 