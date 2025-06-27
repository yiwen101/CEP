"""
Run class for executing experiments on a set of problems
"""

import os
import json
import time
from pathlib import Path
from typing import List, Callable, Dict, Any
from dataclasses import asdict

from .model import (
    Problem, CallData, CallMeta, CallResp, CallMetric,
    RunData, RunMeta, AggregatedRunResults,
    ExperimentMeta, ExperimentDataManager
)
from .analysis import AnalysisUtils
from .evaluator import Evaluator

# Call is an alias for a function that takes query, context and returns CallResp
Call = Callable[[Problem], CallResp]

class Run:
    """Class for executing a single run on a set of problems"""
    
    def __init__(self, experiment_meta: ExperimentMeta, experiment_folder_path: str, 
                 domain: str, model: str, method: str, problems: List[Problem], 
                 call_func: Call, evaluator: Evaluator):
        self.experiment_meta = experiment_meta
        self.experiment_folder_path = Path(experiment_folder_path)
        self.domain = domain
        self.model = model
        self.method = method
        self.problems = problems
        self.call_func = call_func
        self.evaluator = evaluator
        # Auto-generate run metadata
        self.run_meta = ExperimentDataManager.create_run_meta(
            experiment_meta.experiment_id,
            domain,
            method,
            model
        )
        
        # Auto-generate run folder path
        self.run_folder_path = self.experiment_folder_path / "runs" / self.run_meta.run_id
        
        # Create run directory
        self.run_folder_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize calls list for this run
        self.calls = []
    
    def _create_call_meta(self, problem: Problem, problem_index: int) -> CallMeta:
        """Create call metadata for a problem"""
        return CallMeta(
            experiment_id=self.experiment_meta.experiment_id,
            run_id=self.run_meta.run_id,
            sample_id=problem.problem_id or f"problem_{problem_index}",
            question=problem.question,
            gold_answer=problem.gold_answer,
            context_length=len(problem.context),
            sample_type=self.domain,
            sample_level="",
            timestamp=self.run_meta.timestamp
        )
    
    def _execute_single_call(self, problem: Problem, problem_index: int) -> CallData:
        """Execute a single call on a problem"""
        print(f"  Processing problem {problem_index + 1}/{len(self.problems)}")
        
        # Create call metadata
        call_meta = self._create_call_meta(problem, problem_index)
        
        # Execute the call
        start_time = time.time()
        call_resp = self.call_func(problem)
        execution_time = time.time() - start_time
        
        # Update execution time if not provided
        if call_resp.execution_time == 0:
            call_resp.execution_time = execution_time
        
        # Evaluate the call using the provided evaluator
        call_metric = self.evaluator.evaluate(problem.gold_answer, call_resp)
        
        # Create call data
        call_data = ExperimentDataManager.create_call(call_meta, call_resp, call_metric)
        
        # Print progress
        print(f"    Gold: {problem.gold_answer}")
        print(f"    Pred: {call_resp.predicted_answer}")
        print(f"    EM: {call_metric.is_exact_match}, F1: {call_metric.f1:.3f}")
        
        return call_data
    
    def run(self) -> RunData:
        """Execute the run on all problems and save results"""
        print(f"Running {self.method} on {self.domain} with {self.model}")
        print(f"Run ID: {self.run_meta.run_id}")
        print(f"Problems: {len(self.problems)}")
        print(f"Evaluator: {self.evaluator.__class__.__name__}")
        
        # Execute calls on all problems
        for i, problem in enumerate(self.problems):
            call_data = self._execute_single_call(problem, i)
            self.calls.append(call_data)
        
        # Save calls.json (all calls in one file)
        calls_file = self.run_folder_path / "calls.json"
        calls_data = [ExperimentDataManager.to_dict(call) for call in self.calls]
        with open(calls_file, 'w') as f:
            json.dump(calls_data, f, indent=2)
        
        # Create aggregated results from calls
        aggregated_results = ExperimentDataManager.aggregate_calls_to_run_results(self.calls)
        
        # Create run data without the calls array
        run_data = RunData(
            run_meta=self.run_meta,
            aggregated_run_results=aggregated_results,
            calls=[]  # Empty array since calls are stored in calls.json
        )
        
        # Save run.json
        run_file = self.run_folder_path / "run.json"
        with open(run_file, 'w') as f:
            json.dump(ExperimentDataManager.to_dict(run_data), f, indent=2)
        
        print(f"\nRun completed!")
        print(f"Results saved to: {self.run_folder_path}")
        
        # Print summary
        results = aggregated_results
        print(f"Exact Match: {results.avg_is_exact_match:.3f}")
        print(f"F1 Score: {results.avg_f1:.3f}")
        print(f"Precision: {results.avg_precision:.3f}")
        print(f"Recall: {results.avg_recall:.3f}")
        print(f"GT All in Pred: {results.avg_correct:.3f}")
        print(f"Avg Time: {results.avg_execution_time:.2f}s")
        print(f"Avg Tokens: {results.avg_tokens_used:.0f}")
        return run_data 