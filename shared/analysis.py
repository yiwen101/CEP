"""
Analysis utilities for parsing and aggregating experiment data
"""

import json
import os
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import asdict

from .model import (
    CallData, CallMeta, CallResp, CallMetric,
    RunData, RunMeta, AggregatedRunResults,
    ExperimentData, ExperimentMeta, AggregatedExperimentResults
)

class AnalysisUtils:
    """Utility class for analyzing experiment results"""
    
    @staticmethod
    def parse_calls_from_json(calls_file_path: str) -> List[CallData]:
        """Parse a calls.json file and return list of CallData objects"""
        with open(calls_file_path, 'r') as f:
            calls_data = json.load(f)
        
        calls = []
        for call_dict in calls_data:
            # Parse individual components
            call_meta = CallMeta(**call_dict['call_meta'])
            call_resp = CallResp(**call_dict['call_resp'])
            call_metric = CallMetric(**call_dict['call_metric'])
            
            # Create CallData
            call_data = CallData(
                call_meta=call_meta,
                call_resp=call_resp,
                call_metric=call_metric
            )
            calls.append(call_data)
        
        return calls
    
    @staticmethod
    def aggregate_calls_to_run_results(calls: List[CallData]) -> AggregatedRunResults:
        """Aggregate a list of calls into run results"""
        if not calls:
            return AggregatedRunResults(
                avg_is_exact_match=0.0,
                avg_precision=0.0,
                avg_recall=0.0,
                avg_f1=0.0,
                avg_correct=0.0,
                avg_execution_time=0.0,
                avg_tokens_used=0.0,
                sample_count=0
            )
        
        total_calls = len(calls)
        
        # Calculate averages
        avg_is_exact_match = sum(1 for call in calls if call.call_metric.is_exact_match) / total_calls
        avg_precision = sum(call.call_metric.precision for call in calls) / total_calls
        avg_recall = sum(call.call_metric.recall for call in calls) / total_calls
        avg_f1 = sum(call.call_metric.f1 for call in calls) / total_calls
        avg_correct = sum(call.call_metric.correct for call in calls) / total_calls
        avg_execution_time = sum(call.call_resp.execution_time for call in calls) / total_calls
        avg_tokens_used = sum(call.call_resp.tokens_used for call in calls) / total_calls
        
        return AggregatedRunResults(
            avg_is_exact_match=avg_is_exact_match,
            avg_precision=avg_precision,
            avg_recall=avg_recall,
            avg_f1=avg_f1,
            avg_correct=avg_correct,
            avg_execution_time=avg_execution_time,
            avg_tokens_used=avg_tokens_used,
            sample_count=total_calls
        )
    
    @staticmethod
    def create_run_data_from_calls(calls: List[CallData], run_meta: RunMeta) -> RunData:
        """Create RunData from a list of calls and provided run metadata"""
        # Aggregate results
        aggregated_results = AnalysisUtils.aggregate_calls_to_run_results(calls)
        
        # Create run data
        run_data = RunData(
            run_meta=run_meta,
            aggregated_run_results=aggregated_results
        )
        
        return run_data
    
    @staticmethod
    def process_run_directory(run_dir_path: str, run_meta: RunMeta) -> RunData:
        """Process a run directory containing calls.json and create run.json"""
        run_file_path = os.path.join(run_dir_path, "run.json")
        if os.path.exists(run_file_path):
            with open(run_file_path, 'r') as f:
                run_data = json.load(f)
            return RunData(run_meta=RunMeta(**run_data['run_meta']), aggregated_run_results=AggregatedRunResults(**run_data['aggregated_run_results']))
        
        calls_file_path = os.path.join(run_dir_path, "calls.json")
        if not os.path.exists(calls_file_path):
            raise FileNotFoundError(f"calls.json not found in {run_dir_path}")
        
        # Parse calls
        calls = AnalysisUtils.parse_calls_from_json(calls_file_path)
        
        # Create run data using provided metadata
        run_data = AnalysisUtils.create_run_data_from_calls(calls, run_meta)
        
        # Save run.json
        with open(run_file_path, 'w') as f:
            json.dump(asdict(run_data), f, indent=2)
        
        print(f"Created run.json in {run_dir_path}")
        return run_data
    
    @staticmethod
    def process_experiment_directory(experiment_dir_path: str, run_metas: List[RunMeta]) -> List[RunData]:
        """Process an experiment directory with structure:
        experiment/
        ├── runs.json
        └── runs/
            ├── run1/
            │   ├── calls.json
            │   └── run.json
            └── run2/
                ├── calls.json
                └── run.json
        """
        experiment_dir = Path(experiment_dir_path)
        runs_dir = experiment_dir / "runs"
        runs = []
        
        # Create a mapping from run_id to run_meta for easy lookup
        run_meta_map = {meta.run_id: meta for meta in run_metas}
        
        # Process each run directory under runs/
        if runs_dir.exists():
            for run_subdir in runs_dir.iterdir():
                if run_subdir.is_dir():
                    run_id = run_subdir.name
                    
                    # Get run_meta from provided list
                    if run_id in run_meta_map:
                        run_meta = run_meta_map[run_id]
                        run_data = AnalysisUtils.process_run_directory(str(run_subdir), run_meta)
                        runs.append(run_data)
                    else:
                        print(f"No run_meta found for {run_id}, skipping...")
                        continue
        else:
            print(f"runs/ directory not found in {experiment_dir_path}")
        
        # Save runs.json in the experiment directory
        runs_file_path = experiment_dir / "runs.json"
        runs_data = [asdict(run) for run in runs]
        with open(runs_file_path, 'w') as f:
            json.dump(runs_data, f, indent=2)
        
        print(f"Created runs.json in {experiment_dir_path} with {len(runs)} runs")
        
        return runs
    
    @staticmethod
    def print_run_summary(run_data: RunData):
        """Print a summary of run results"""
        results = run_data.aggregated_run_results
        print(f"\nRun: {run_data.run_meta.method} ({run_data.run_meta.model})")
        print(f"Dataset: {run_data.run_meta.dataset}")
        print(f"Samples: {results.sample_count}")
        print(f"Exact Match: {results.avg_is_exact_match:.3f}")
        print(f"F1 Score: {results.avg_f1:.3f}")
        print(f"Precision: {results.avg_precision:.3f}")
        print(f"Recall: {results.avg_recall:.3f}")
        print(f"GT All in Pred: {results.avg_correct:.3f}")
        print(f"Avg Time: {results.avg_execution_time:.2f}s")
        print(f"Avg Tokens: {results.avg_tokens_used:.0f}")
    
    @staticmethod
    def print_experiment_summary(runs: List[RunData]):
        """Print a summary of experiment results"""
        if not runs:
            print("No runs found")
            return
            
        print(f"\nExperiment: {runs[0].run_meta.experiment_id}")
        print(f"Dataset: {runs[0].run_meta.dataset}")
        print(f"Total Runs: {len(runs)}")
        
        print("\n" + "="*120)
        print(f"{'Method':<30} {'Model':<15} {'EM':<8} {'F1':<8} {'Precision':<10} {'Recall':<8} {'GT_all_in_pred':<12} {'Time(s)':<10} {'Tokens':<10}")
        print("-"*120)
        
        for run in runs:
            results = run.aggregated_run_results
            print(f"{run.run_meta.method:<30} {run.run_meta.model:<15} {results.avg_is_exact_match:<8.3f} {results.avg_f1:<8.3f} "
                  f"{results.avg_precision:<10.3f} {results.avg_recall:<8.3f} {results.avg_correct:<12.3f} "
                  f"{results.avg_execution_time:<10.2f} {results.avg_tokens_used:<10.0f}")
