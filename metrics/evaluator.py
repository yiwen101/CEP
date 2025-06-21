import re
import string
import collections
from typing import List, Dict, Tuple
import numpy as np

# Assuming ExperimentResult is defined elsewhere and imported, or passed as a Dict
# from ..cep_hotpot_experiment import ExperimentResult 

class HotPotQAEvaluator:
    """Handles evaluation for HotPotQA dataset."""

    def normalize_answer(self, s: str) -> str:
        """Normalize answer for evaluation (from HotPotQA evaluation script)"""
        def remove_articles(text):
            return re.sub(r'\b(a|an|the)\b', ' ', text)
        
        def white_space_fix(text):
            return ' '.join(text.split())
        
        def remove_punc(text):
            exclude = set(string.punctuation)
            return ''.join(ch for ch in text if ch not in exclude)
        
        def lower(text):
            return text.lower()
        
        return white_space_fix(remove_articles(remove_punc(lower(s))))

    def get_tokens(self, s: str) -> List[str]:
        """Get tokens from normalized string"""
        if not s:
            return []
        return self.normalize_answer(s).split()

    def compute_exact(self, a_gold: str, a_pred: str) -> int:
        """Compute exact match score"""
        return int(self.normalize_answer(a_gold) == self.normalize_answer(a_pred))

    def compute_f1(self, a_gold: str, a_pred: str) -> Tuple[float, float, float]:
        """Compute F1 score with precision and recall"""
        gold_toks = self.get_tokens(a_gold)
        pred_toks = self.get_tokens(a_pred)
        
        if not gold_toks and not pred_toks:
            return 1.0, 1.0, 1.0

        if not gold_toks or not pred_toks:
            return 0.0, 0.0, 0.0

        common = collections.Counter(gold_toks) & collections.Counter(pred_toks)
        num_same = sum(common.values())
        
        if num_same == 0:
            return 0.0, 0.0, 0.0
        
        precision = 1.0 * num_same / len(pred_toks)
        recall = 1.0 * num_same / len(gold_toks)
        f1 = (2 * precision * recall) / (precision + recall)
        return f1, precision, recall

    def calculate_metrics(self, results: List[Dict]) -> Dict[str, float]:
        """Calculate evaluation metrics using correct HotPotQA evaluation"""
        total = len(results)
        if total == 0:
            return {}
            
        exact_matches = 0
        f1_scores = []
        precision_scores = []
        recall_scores = []
        llm_correctness_scores = []
        
        for result_dict in results:
            result = collections.namedtuple("ExperimentResult", result_dict.keys())(*result_dict.values())
            
            # Compute exact match
            em = self.compute_exact(result.gold_answer, result.predicted_answer)
            exact_matches += em
            
            # Compute F1, precision, recall
            f1, precision, recall = self.compute_f1(result.gold_answer, result.predicted_answer)
            f1_scores.append(f1)
            precision_scores.append(precision)
            recall_scores.append(recall)
            
            # LLM correctness
            if hasattr(result, 'llm_correctness') and result.llm_correctness is not None:
                llm_correctness_scores.append(100 if result.llm_correctness else 0.0)
        
        avg_execution_time = np.mean([r['execution_time'] for r in results])
        avg_tokens_used = np.mean([r['tokens_used'] for r in results])
        
        metrics = {
            'exact_match': exact_matches / total,
            'f1_score': np.mean(f1_scores),
            'precision': np.mean(precision_scores),
            'recall': np.mean(recall_scores),
            'avg_execution_time': avg_execution_time,
            'avg_tokens_used': avg_tokens_used,
            'total_samples': total
        }
        
        # Add LLM correctness if available
        if llm_correctness_scores:
            metrics['llm_correctness'] = np.mean(llm_correctness_scores)
        
        return metrics 