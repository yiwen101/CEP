"""
Evaluator interface and implementations for evaluating CallResp and returning CallMetric
"""
import re
import string
import collections
import logging
from typing import Optional, Tuple, List, Dict, Any
from .model import CallResp, CallMetric
from .utils.grader import check_is_correct

logger = logging.getLogger(__name__)

class Evaluator:
    """HotPotQA-style evaluator with more sophisticated normalization"""
    
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
    
    def get_tokens(self, s: str) -> list:
        """Get tokens from normalized string"""
        if not s:
            return []
        return self.normalize_answer(s).split()
    
    def compute_exact(self, a_gold: str, a_pred: str) -> int:
        """Compute exact match score"""
        return int(self.normalize_answer(a_gold) == self.normalize_answer(a_pred))
    
    def compute_f1(self, a_gold: str, a_pred: str) -> tuple[float, float, float, float]:
        """Compute F1 score with precision and recall"""
        gold_toks = self.get_tokens(a_gold)
        pred_toks = self.get_tokens(a_pred)
        
        if not gold_toks and not pred_toks:
            return 1.0, 1.0, 1.0, 1.0

        if not gold_toks or not pred_toks:
            return 0.0, 0.0, 0.0, 0.0

        common = collections.Counter(gold_toks) & collections.Counter(pred_toks)
        num_same = sum(common.values())
        
        if num_same == 0:
            return 0.0, 0.0, 0.0, 0.0
        
        precision = 1.0 * num_same / len(pred_toks)
        recall = 1.0 * num_same / len(gold_toks)
        f1 = (2 * precision * recall) / (precision + recall)
        correct = 1.0 if all(token in pred_toks for token in gold_toks) else 0.0
        return f1, precision, recall, correct
    
    def evaluate(self, gold_answer: str, call_resp: CallResp) -> CallMetric:
        """Evaluate using HotPotQA-style metrics"""
        predicted_answer = call_resp.predicted_answer
        
        # Compute exact match
        is_exact_match = bool(self.compute_exact(gold_answer, predicted_answer))
        
        # Compute F1, precision, recall
        f1, precision, recall, correct = self.compute_f1(gold_answer, predicted_answer)
        
        return CallMetric(
            is_exact_match=is_exact_match,
            precision=precision,
            recall=recall,
            f1=f1,
            correct=correct,
            llm_correctness=None
        )

class MathEvaluator(Evaluator):
    """Math-specific evaluator that extends base evaluator with DEER's capabilities"""

    def normalize_math_answer(self, answer: str) -> str:
        """Normalize mathematical answer for comparison"""
        if not answer:
            return ""
            
        # Convert fractions to decimals
        if '/' in answer:
            try:
                num, denom = map(float, answer.split('/'))
                answer = str(num / denom)
            except:
                pass
                
        # Remove whitespace and convert to lowercase
        answer = answer.strip().lower()
        
        # Replace unicode minus with hyphen
        answer = answer.replace('−', '-')
        
        # Remove unnecessary .0 from integers
        if answer.endswith('.0'):
            answer = answer[:-2]
            
        return answer

    def check_math_equivalence(self, a: str, b: str) -> bool:
        """Check if two mathematical expressions are equivalent"""
        a_norm = self.normalize_math_answer(a)
        b_norm = self.normalize_math_answer(b)
        
        if not a_norm or not b_norm:
            return False
            
        try:
            # Convert to float for numerical comparison
            a_val = float(a_norm)
            b_val = float(b_norm)
            # Use small epsilon for float comparison
            return abs(a_val - b_val) < 1e-6
        except:
            # If conversion fails, fall back to string comparison
            return a_norm == b_norm

    def evaluate(self, gold_answer: str, call_resp: CallResp) -> CallMetric:
        """Evaluate mathematical answers using DEER's evaluation logic"""
        predicted_answer = call_resp.predicted_answer
        
        # Check exact match using math-specific comparison
        correct = check_is_correct(predicted_answer,gold_answer)
        
        return CallMetric(
            is_exact_match=False,
            precision=0,
            recall=0,
            f1=0,
            correct=1.0 if correct else 0.0,
            llm_correctness=None
        )
