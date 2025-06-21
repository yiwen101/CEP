"""
Experiment framework for running CEP experiments across multiple datasets and models
"""

import os
import json
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Callable, Optional
from pathlib import Path
from dataclasses import asdict

from .model import Problem, ExperimentMeta, ExperimentDataManager
from .run import Run, Call
from .evaluator import Evaluator

class Dataset(ABC):
    """Abstract interface for datasets"""
    
    @abstractmethod
    def get_dataset_name(self) -> str:
        """Return the name of this dataset"""
        pass
    
    @abstractmethod
    def get_domains(self) -> List[str]:
        """Return list of available domains in this dataset"""
        pass
    
    @abstractmethod
    def get_problems(self, domain: str, max_samples: int) -> List[Problem]:
        """Return list of problems for a given domain"""
        pass

class CallBuilder(ABC):
    """Abstract interface for call builders"""
    
    @abstractmethod
    def build_calls(self, model: str, domain: str) -> Dict[str, Call]:
        """Build a map of method names to call functions for a given model and domain"""
        pass

class Experiment:
    """Main experiment class for running experiments across multiple domains and models"""
    
    def __init__(self, dataset: Dataset, call_builder: CallBuilder, output_dir: str = "results"):
        """
        Initialize experiment
        
        Args:
            dataset: Dataset interface that provides problems
            call_builder: CallBuilder that creates calls for model/domain combinations
            output_dir: Base directory for experiment outputs
        """
        self.dataset = dataset
        self.call_builder = call_builder
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Validate that all domains exist in the dataset
        available_domains = dataset.get_domains()
        print(f"Available domains: {available_domains}")
    
    def run(self, domains: List[str], models: List[str], max_samples: int = 50) -> Dict[str, Any]:
        """
        Run experiments across specified domains and models
        
        Args:
            domains: List of domains to run experiments on
            models: List of model names to test
            max_samples: Maximum number of samples per domain
            
        Returns:
            Dictionary with experiment results summary
        """
        print(f"Starting experiment with {len(domains)} domains and {len(models)} models")
        print(f"Domains: {domains}")
        print(f"Models: {models}")
        print(f"Max samples per domain: {max_samples}")
        print(f"Output directory: {self.output_dir}")
        
        # Create experiment metadata with auto-generated ID
        dataset_name = self.dataset.get_dataset_name()
        experiment_meta = ExperimentDataManager.create_experiment_meta(dataset_name)
        experiment_folder = self.output_dir / experiment_meta.experiment_id
        experiment_folder.mkdir(parents=True, exist_ok=True)
        
        # Create runs subdirectory
        runs_folder = experiment_folder / "runs"
        runs_folder.mkdir(parents=True, exist_ok=True)
        
        # Save experiment metadata
        experiment_meta_file = experiment_folder / "experiment_meta.json"
        with open(experiment_meta_file, 'w') as f:
            json.dump(ExperimentDataManager.to_dict(experiment_meta), f, indent=2)
        
        all_results = {}
        
        # Run experiments for each domain and model combination
        for domain in domains:
            print(f"\n{'='*80}")
            print(f"PROCESSING DOMAIN: {domain.upper()}")
            print(f"{'='*80}")
            
            # Get problems for this domain
            try:
                problems = self.dataset.get_problems(domain, max_samples)
                print(f"Loaded {len(problems)} problems for domain {domain}")
            except Exception as e:
                print(f"Error loading problems for domain {domain}: {e}")
                continue
            
            # Create evaluator for this domain
            evaluator = Evaluator()
            
            domain_results = {}
            
            for model in models:
                print(f"\n--- Running {model} on {domain} ---")
                
                # Build calls for this model and domain
                calls = self.call_builder.build_calls(model, domain)
                print(f"  Built {len(calls)} call functions for {model} on {domain}")
                
                model_results = {}
                
                for method_name, call_func in calls.items():
                    print(f"  Running {method_name}...")
                    
                    try:
                        # Create and run the experiment
                        run = Run(
                            experiment_meta=experiment_meta,
                            experiment_folder_path=str(experiment_folder),
                            domain=domain,
                            model=model,
                            method=method_name,
                            problems=problems,
                            call_func=call_func,
                            evaluator=evaluator
                        )
                        
                        run_data = run.run()
                        
                        # Store results
                        model_results[method_name] = {
                            'run_id': run_data.run_meta.run_id,
                            'metrics': {
                                'exact_match': run_data.aggregated_run_results.avg_is_exact_match,
                                'f1': run_data.aggregated_run_results.avg_f1,
                                'precision': run_data.aggregated_run_results.avg_precision,
                                'recall': run_data.aggregated_run_results.avg_recall,
                                'ground_truth_all_in_pred': run_data.aggregated_run_results.avg_ground_truth_all_in_pred,
                                'avg_execution_time': run_data.aggregated_run_results.avg_execution_time,
                                'avg_tokens_used': run_data.aggregated_run_results.avg_tokens_used,
                                'sample_count': run_data.aggregated_run_results.sample_count
                            }
                        }
                        
                        print(f"    Completed {method_name}: EM={run_data.aggregated_run_results.avg_is_exact_match:.3f}, F1={run_data.aggregated_run_results.avg_f1:.3f}")
                        
                    except Exception as e:
                        print(f"    Error running {method_name}: {e}")
                        model_results[method_name] = {'error': str(e)}
                
                domain_results[model] = model_results
            
            all_results[domain] = domain_results
        
        # Save overall experiment summary
        summary_file = experiment_folder / "experiment_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(all_results, f, indent=2)
        
        # Print final summary
        self._print_experiment_summary(all_results)
        
        print(f"\nExperiment completed!")
        print(f"Results saved to: {experiment_folder}")
        print(f"Summary saved to: {summary_file}")
        
        return all_results
    
    def _print_experiment_summary(self, results: Dict[str, Any]):
        """Print a formatted summary of experiment results"""
        print("\n" + "="*120)
        print("EXPERIMENT SUMMARY")
        print("="*120)
        
        for domain, domain_results in results.items():
            print(f"\nDOMAIN: {domain.upper()}")
            print("-" * 80)
            
            # Print header
            print(f"{'Model':<20} {'Method':<25} {'EM':<8} {'F1':<8} {'Precision':<10} {'Recall':<8} {'GT_all_in_pred':<12} {'Time(s)':<10} {'Tokens':<10}")
            print("-" * 80)
            
            for model, model_results in domain_results.items():
                for method, method_data in model_results.items():
                    if 'error' in method_data:
                        print(f"{model:<20} {method:<25} {'ERROR':<8}")
                    else:
                        metrics = method_data['metrics']
                        print(f"{model:<20} {method:<25} {metrics['exact_match']:<8.3f} "
                              f"{metrics['f1']:<8.3f} {metrics['precision']:<10.3f} "
                              f"{metrics['recall']:<8.3f} {metrics['ground_truth_all_in_pred']:<12.3f} "
                              f"{metrics['avg_execution_time']:<10.2f} {metrics['avg_tokens_used']:<10.0f}")
        
        print("\n" + "="*120) 