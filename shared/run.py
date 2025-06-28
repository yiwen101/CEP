"""
Run class for executing experiments on a set of problems
"""

import os
import json
import time
from pathlib import Path
from typing import List, Callable, Dict, Any
from dataclasses import asdict
from concurrent.futures import ThreadPoolExecutor, as_completed

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

    def _execute_call(self, problem: Problem, problem_index: int) -> CallData:
        """Execute a single call on a problem"""
        print(f"  Processing problem {problem_index + 1}/{len(self.problems)} ({self.method})")
        
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
        #print(f"    Gold: {problem.gold_answer}")
        #print(f"    Pred: {call_resp.predicted_answer}")
        #print(f"    EM: {call_metric.is_exact_match}, F1: {call_metric.f1:.3f}")
        
        return call_data

    def _execute_calls(self, problems: List[Problem], sequential: bool = False) -> List[CallData]:
        """Execute calls on a list of problems, either sequentially or in parallel"""
        # Initialize calls.json
        calls_file = self.run_folder_path / "calls.json"
        with open(calls_file, 'w') as f:
            json.dump([], f)

        if sequential:
            # Run calls sequentially
            results = [self._execute_call(problem, i) for i, problem in enumerate(problems)]
        else:
            # Run calls in parallel with n=8 workers
            with ThreadPoolExecutor(max_workers=25) as executor:
                future_to_problem = {
                    executor.submit(self._execute_call, problem, i): (problem, i)
                    for i, problem in enumerate(problems)
                }
                
                results = []
                for future in as_completed(future_to_problem):
                    problem, i = future_to_problem[future]
                    try:
                        call_data = future.result()
                        results.append(call_data)
                    except Exception as e:
                        print(f"Error processing problem {i} for {self.method}: {e}")
        
        # Sort results by problem index to maintain order
        results.sort(key=lambda x: int(x.call_meta.sample_id.split('_')[-1]) if '_' in x.call_meta.sample_id else 0)
        
        # Save all call data at once
        calls_data = [ExperimentDataManager.to_dict(call_data) for call_data in results]
        with open(calls_file, 'w') as f:
            json.dump(calls_data, f, indent=2)
            
        return results
    
    def run(self) -> RunData:
        """Execute the run on all problems and save results"""
        print(f"Running {self.method} on {self.domain} with {self.model}")
        print(f"Run ID: {self.run_meta.run_id}")
        print(f"Problems: {len(self.problems)}")
        print(f"Evaluator: {self.evaluator.__class__.__name__}")
        
        # Execute calls on all problems
        self.calls = self._execute_calls(self.problems)
        run_data = RunData(
            run_meta=self.run_meta,
            aggregated_run_results= ExperimentDataManager.aggregate_calls_to_run_results(self.calls),
            calls=[]
        )
        
        # Save run data
        run_file = self.run_folder_path / "run.json"
        with open(run_file, 'w') as f:
            json.dump(ExperimentDataManager.to_dict(run_data), f, indent=2)
        
        return run_data 