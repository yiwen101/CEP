"""
Evaluator interface and implementations for evaluating CallResp and returning CallMetric
"""
from .model import CallResp, CallMetric

class Evaluator:
    """HotPotQA-style evaluator with more sophisticated normalization"""
    
    def normalize_answer(self, s: str) -> str:
        """Normalize answer for evaluation (from HotPotQA evaluation script)"""
        import re
        import string
        
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
        import collections
        
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
        ground_truth_all_in_pred = 1.0 if all(token in pred_toks for token in gold_toks) else 0.0
        return f1, precision, recall, ground_truth_all_in_pred
    
    def evaluate(self, gold_answer: str, call_resp: CallResp) -> CallMetric:
        """Evaluate using HotPotQA-style metrics"""
        predicted_answer = call_resp.predicted_answer
        
        # Compute exact match
        is_exact_match = bool(self.compute_exact(gold_answer, predicted_answer))
        
        # Compute F1, precision, recall
        f1, precision, recall, ground_truth_all_in_pred = self.compute_f1(gold_answer, predicted_answer)
        
        return CallMetric(
            is_exact_match=is_exact_match,
            precision=precision,
            recall=recall,
            f1=f1,
            ground_truth_all_in_pred=ground_truth_all_in_pred,
            llm_correctness=None
        )
