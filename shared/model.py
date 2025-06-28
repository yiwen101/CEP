"""
Data models for experiment management
"""

import json
import time
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional, Type, TypeVar
from datetime import datetime

T = TypeVar('T')

@dataclass
class Problem:
    """Represents a single problem with question, context, and gold answer"""
    question: str
    context: str
    gold_answer: str
    problem_id: str
    choices: List[str]

@dataclass
class CallMeta:
    """Metadata for a single API call"""
    experiment_id: str
    run_id: str
    sample_id: str
    question: str
    gold_answer: str
    context_length: int
    sample_type: str = ""
    sample_level: str = ""
    timestamp: str = ""

@dataclass
class CallResp:
    """Response data from a single API call"""
    predicted_answer: str
    execution_time: float
    tokens_used: int
    chat_history: List[Dict[str, str]]
    elaboration: Optional[str] = None
    reasoning_tokens: int = 0

@dataclass
class CallMetric:
    """Evaluation metrics for a single call"""
    is_exact_match: bool
    precision: float
    recall: float
    f1: float
    correct: float
    step_count: int

@dataclass
class CallData:
    """Complete data for a single API call"""
    call_meta: CallMeta
    call_resp: CallResp
    call_metric: CallMetric

@dataclass
class RunMeta:
    """Metadata for a run (set of calls with same method/model)"""
    experiment_id: str
    run_id: str
    domain: str
    method: str
    model: str
    timestamp: str = ""

@dataclass
class AggregatedRunResults:
    """Aggregated metrics across all calls in a run"""
    avg_is_exact_match: float
    avg_precision: float
    avg_recall: float
    avg_f1: float
    avg_correct: float
    avg_execution_time: float
    avg_tokens_used: float
    avg_reasoning_tokens: float
    avg_step_count: float
    sample_count: int

@dataclass
class RunData:
    """Complete data for a run"""
    run_meta: RunMeta
    aggregated_run_results: AggregatedRunResults
    calls: List[CallData] = field(default_factory=list)

@dataclass
class ExperimentMeta:
    """Metadata for an experiment"""
    experiment_id: str # Format: "{dataset}-{timestamp}"
    dataset: str

@dataclass
class AggregatedExperimentResults:
    """Aggregated results across all runs in an experiment"""
    runs: List[AggregatedRunResults] = field(default_factory=list)

@dataclass
class ExperimentData:
    """Complete data for an experiment"""
    experiment_meta: ExperimentMeta
    aggregated_experiment_results: AggregatedExperimentResults
    runs: List[RunData] = field(default_factory=list)

@dataclass
class ElaborationData:
    """Complete data for an elaboration"""
    elaboration: str
    elaboration_history: List[Dict[str, str]]
    tokens_used: int
    execution_time: float

class ExperimentDataManager:
    """Utility class for creating, aggregating, and serializing experiment data"""
    
    @staticmethod
    def create_experiment_meta(dataset: str, with_cot: bool, custom_id: str = None) -> ExperimentMeta:
        """Create experiment metadata with auto-generated ID or custom ID"""
        if custom_id:
            experiment_id = custom_id
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            cot = "-cot" if with_cot else ""
            experiment_id = f"{dataset}{cot}-{timestamp}"
        return ExperimentMeta(
            experiment_id=experiment_id,
            dataset=dataset
        )
    
    @staticmethod
    def create_run_meta(experiment_id: str, domain: str, method: str, model: str) -> RunMeta:
        """Create run metadata with auto-generated ID"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_id = f"{domain}_{model}_{method}"  # Removed timestamp for clarity
        return RunMeta(
            experiment_id=experiment_id,
            run_id=run_id,
            domain=domain,
            method=method,
            model=model,
            timestamp=timestamp
        )
    
    @staticmethod
    def create_call(call_meta: CallMeta, call_resp: CallResp, call_metric: CallMetric) -> CallData:
        """Create a complete call data object"""
        return CallData(
            call_meta=call_meta,
            call_resp=call_resp,
            call_metric=call_metric
        )
    
    @staticmethod
    def aggregate_calls_to_run_results(calls: List[CallData]) -> AggregatedRunResults:
        """Aggregate metrics from a list of calls into run results"""
        if not calls:
            return AggregatedRunResults(
                avg_is_exact_match=0.0,
                avg_precision=0.0,
                avg_recall=0.0,
                avg_f1=0.0,
                avg_correct=0.0,
                avg_execution_time=0.0,
                avg_tokens_used=0.0,
                avg_reasoning_tokens=0.0,
                avg_step_count=0.0,
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
        avg_reasoning_tokens = sum(call.call_resp.reasoning_tokens for call in calls) / total_calls
        avg_step_count = sum(call.call_metric.step_count for call in calls) / total_calls
        
        return AggregatedRunResults(
            avg_is_exact_match=avg_is_exact_match,
            avg_precision=avg_precision,
            avg_recall=avg_recall,
            avg_f1=avg_f1,
            avg_correct=avg_correct,
            avg_execution_time=avg_execution_time,
            avg_tokens_used=avg_tokens_used,
            avg_reasoning_tokens=avg_reasoning_tokens,
            avg_step_count=avg_step_count,
            sample_count=total_calls
        )
    
    @staticmethod
    def to_dict(obj: Any) -> Dict[str, Any]:
        """Convert a dataclass object to dictionary"""
        return asdict(obj)
    
    @staticmethod
    def from_dict(data: Dict[str, Any], cls: Type[T]) -> T:
        """Convert dictionary back to dataclass object"""
        return cls(**data)
    
    @staticmethod
    def save_to_json(obj: Any, filepath: str):
        """Save a dataclass object to JSON file"""
        with open(filepath, 'w') as f:
            json.dump(asdict(obj), f, indent=2)
    
    @staticmethod
    def load_from_json(filepath: str, cls: Type[T]) -> T:
        """Load a dataclass object from JSON file"""
        with open(filepath, 'r') as f:
            data = json.load(f)
        return cls(**data)