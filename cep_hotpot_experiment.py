#!/usr/bin/env python3
"""
Context Elaboration Prompt (CEP) Experiment on HotPotQA
Testing the effectiveness of structured context elaboration for multi-hop reasoning

This implementation tests the methods proposed in the research proposal:
- Method 1: Real-Time In-Context Elaboration (ICE)
- Method 2: In-Context Elaboration as context augmentation
- Baseline methods for comparison
"""

import json
import os
import time
import argparse
import random
import re
import string
import collections
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
from openai import OpenAI
import numpy as np
from tqdm import tqdm
import logging
from dotenv import load_dotenv

from metrics.evaluator import HotPotQAEvaluator

# Load environment variables from .env file
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class HotPotQASample:
    """Data structure for a HotPotQA sample"""
    _id: str
    question: str
    answer: str
    supporting_facts: List[List[str]]
    context: List[List[str]]  # List of [title, sentences]
    type: Optional[str] = None
    level: Optional[str] = None

@dataclass
class ExperimentResult:
    """Results from a single experiment"""
    method_name: str
    sample_id: str
    question: str
    gold_answer: str
    predicted_answer: str
    elaboration: Optional[str] = None
    reasoning: Optional[str] = None
    execution_time: float = 0.0
    tokens_used: int = 0
    llm_correctness: Optional[bool] = None

class CEPHotPotExperiment:
    """Main experiment class for testing CEP methods on HotPotQA"""
    
    def __init__(self, openai_api_key: str, model: str = "gpt-3.5-turbo", max_samples: int = 100):
        self.client = OpenAI(api_key=openai_api_key)
        self.model = model
        self.max_samples = max_samples
        self.evaluator = HotPotQAEvaluator()
        
        # Context Elaboration Prompts (CEPs) based on research proposal
        self.ceps = {
            "understand": [
                "Paraphrase the provided information in your own words",
                "Summarize the given text in a clear and concise manner"
            ],
            "connect": [
                "What does this information remind you of? Briefly explain the connection.",
                "How does this information relate to other facts or concepts you know?"
            ],
            "query": [
                "What do you find to be the most surprising or interesting piece of information?",
                "Formulate two insightful questions that are raised by the text"
            ],
            "application": [
                "What can you deduce from the given information?",
                "Formulate two insightful questions that are answered by the information given"
            ],
            "comprehensive": [
                """Please elaborate on the provided context by:
                1. Paraphrase the provided information in your own words
                2. What does this information remind you of? Briefly explain the connection.
                3. Formulate two insightful questions that are raised by the text
                4. What can you deduce from the given information?"""
            ]
        }
        
        # Baseline prompts
        self.baseline_prompts = {
            "direct": "Answer the following question based on the provided context:",
            "cot": "Let's approach this step by step. First, let me understand the question and the context, then I'll work through the reasoning to find the answer.",
            "cot_explicit": """Let's solve this step by step:

1. First, let me understand what the question is asking
2. Then, I'll identify the relevant information from the context
3. I'll trace through the reasoning needed to connect the information
4. Finally, I'll provide the answer

Question: {question}

Context:
{context}

Let me work through this:""",
            "joint_recall": 0
        }
    
    def load_hotpot_data(self, data_file: str) -> List[HotPotQASample]:
        """Load HotPotQA data from JSON file"""
        logger.info(f"Loading data from {data_file}")
        with open(data_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        samples = []
        for item in data[:self.max_samples]:
            sample = HotPotQASample(
                _id=item['_id'],
                question=item['question'],
                answer=item['answer'],
                supporting_facts=item['supporting_facts'],
                context=item['context'],
                type=item.get('type'),
                level=item.get('level')
            )
            samples.append(sample)
        
        logger.info(f"Loaded {len(samples)} samples")
        return samples
    
    def format_context(self, context: List[List[str]]) -> str:
        """Format context for prompt"""
        formatted_context = []
        for title, sentences in context:
            formatted_context.append(f"Title: {title}")
            formatted_context.extend(sentences)
            formatted_context.append("")
        return "\n".join(formatted_context)
    
    def call_openai_with_history(self, messages: List[Dict[str, str]], max_tokens: int = 1000, temperature: float = 0.0, 
                                model: str | None = None, max_retries: int = 10, base_delay: float = 1.0) -> Tuple[str, int, float]:
        """Make a call to OpenAI API with conversation history and return response with token count"""
        for attempt in range(max_retries):
            try:
                start_time = time.time()
                response = self.client.chat.completions.create(
                    model=model if model else self.model,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature
                )
                execution_time = time.time() - start_time
                
                content = response.choices[0].message.content
                tokens_used = response.usage.total_tokens
                
                return content, tokens_used, execution_time
                
            except Exception as e:
                if attempt == max_retries - 1:  # Last attempt
                    logger.error(f"OpenAI API call failed after {max_retries} attempts: {e}")
                    return f"Error: {e}", 0, 0.0
                
                # Calculate delay with exponential backoff and jitter
                delay = base_delay * (2 ** attempt) + random.uniform(0, 1)
                logger.warning(f"API call failed (attempt {attempt + 1}/{max_retries}), retrying in {delay:.2f}s: {e}")
                time.sleep(delay)
    
    def evaluate_correctness_with_llm(self, question: str, gold_answer: str, predicted_answer: str) -> bool:
        """Evaluate correctness using LLM with simple prompt"""
        messages = [
            {"role": "user", "content": f"""Question: {question}
Gold Answer: {gold_answer}
Predicted Answer: {predicted_answer}

Is the predicted answer correct? Respond with only "correct" or "wrong"."""}
        ]
        
        response, _, _ = self.call_openai_with_history(messages, max_tokens=10, temperature=0.0)
        return response.strip().lower() == "correct"
    
    def cep_method(self, sample: HotPotQASample, cep_category: str, prompt_index: int = 0) -> Tuple[ExperimentResult, ExperimentResult]:
        """Unified CEP method - returns both augmentation and history variants, reusing elaboration"""
        cep_prompts = self.ceps[cep_category]
        cep_prompt = cep_prompts[prompt_index]
        
        # Turn 1: Context + CEP → Elaboration (same for both variants)
        messages = [
            {"role": "user", "content": f"""Context:
{self.format_context(sample.context)}

{cep_prompt}"""}
        ]
        
        elaboration, tokens_1, time_1 = self.call_openai_with_history(messages)
        
        # Variant 1: Augmentation (without history, modified query)
        augmentation_query = f"""Original Context:
{self.format_context(sample.context)}

Elaborated Context:
{elaboration}

Question: {sample.question}

Please provide your answer:"""

        augmentation_messages = [
            {"role": "user", "content": augmentation_query}
        ]
        
        augmentation_answer, tokens_2_aug, time_2_aug = self.call_openai_with_history(augmentation_messages)
        
        # Variant 2: Message History (with full conversation history)
        history_messages = messages.copy()
        history_messages.extend([
            {"role": "assistant", "content": elaboration},
            {"role": "user", "content": f"""Answer the following question:

Question: {sample.question}

Please provide your answer:"""}
        ])
        
        history_answer, tokens_2_hist, time_2_hist = self.call_openai_with_history(history_messages)
        
        # Evaluate correctness for both variants
        augmentation_correctness = self.evaluate_correctness_with_llm(
            sample.question, sample.answer, augmentation_answer
        )
        
        history_correctness = self.evaluate_correctness_with_llm(
            sample.question, sample.answer, history_answer
        )
        
        prompt_suffix = f"_{prompt_index}" if len(cep_prompts) > 1 else ""
        
        # Create results for both variants
        augmentation_result = ExperimentResult(
            method_name=f"cep_augmentation_{cep_category}{prompt_suffix}",
            sample_id=sample._id,
            question=sample.question,
            gold_answer=sample.answer,
            predicted_answer=augmentation_answer,
            elaboration=elaboration,
            execution_time=time_1 + time_2_aug,
            tokens_used=tokens_1 + tokens_2_aug,
            llm_correctness=augmentation_correctness
        )
        
        history_result = ExperimentResult(
            method_name=f"cep_history_{cep_category}{prompt_suffix}",
            sample_id=sample._id,
            question=sample.question,
            gold_answer=sample.answer,
            predicted_answer=history_answer,
            elaboration=elaboration,
            execution_time=time_1 + time_2_hist,
            tokens_used=tokens_1 + tokens_2_hist,
            llm_correctness=history_correctness
        )
        
        return augmentation_result, history_result
    
    def baseline_direct(self, sample: HotPotQASample) -> ExperimentResult:
        """Baseline: Direct question answering without elaboration"""
        messages = [
            {"role": "user", "content": f"""Context:
{self.format_context(sample.context)}

Question: {sample.question}

Please provide your answer:"""}
        ]
        
        predicted_answer, tokens_used, execution_time = self.call_openai_with_history(messages)
        
        # Evaluate correctness with LLM
        llm_correctness = self.evaluate_correctness_with_llm(
            sample.question, sample.answer, predicted_answer
        )
        
        return ExperimentResult(
            method_name="baseline_direct",
            sample_id=sample._id,
            question=sample.question,
            gold_answer=sample.answer,
            predicted_answer=predicted_answer,
            elaboration="",
            execution_time=execution_time,
            tokens_used=tokens_used,
            llm_correctness=llm_correctness
        )
    
    def baseline_cot(self, sample: HotPotQASample) -> ExperimentResult:
        """Baseline: Chain-of-Thought reasoning"""
        messages = [
            {"role": "user", "content": f"""Context:
{self.format_context(sample.context)}

Question: {sample.question}

Please think step by step and then provide your answer:"""}
        ]
        
        predicted_answer, tokens_used, execution_time = self.call_openai_with_history(messages)
        
        # Evaluate correctness with LLM
        llm_correctness = self.evaluate_correctness_with_llm(
            sample.question, sample.answer, predicted_answer
        )
        
        return ExperimentResult(
            method_name="baseline_cot",
            sample_id=sample._id,
            question=sample.question,
            gold_answer=sample.answer,
            predicted_answer=predicted_answer,
            elaboration="",
            execution_time=execution_time,
            tokens_used=tokens_used,
            llm_correctness=llm_correctness
        )
    
    def normalize_answer(self, answer: str) -> str:
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
        
        return white_space_fix(remove_articles(remove_punc(lower(answer))))
    
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
        common = collections.Counter(gold_toks) & collections.Counter(pred_toks)
        num_same = sum(common.values())
        
        if len(gold_toks) == 0 or len(pred_toks) == 0:
            # If either is no-answer, then F1 is 1 if they agree, 0 otherwise
            return int(gold_toks == pred_toks), int(gold_toks == pred_toks), int(gold_toks == pred_toks)
        
        if num_same == 0:
            return 0.0, 0.0, 0.0
        
        precision = 1.0 * num_same / len(pred_toks)
        recall = 1.0 * num_same / len(gold_toks)
        f1 = (2 * precision * recall) / (precision + recall)
        return f1, precision, recall
    
    def calculate_metrics(self, results: List[ExperimentResult]) -> Dict[str, float]:
        """Calculate evaluation metrics using correct HotPotQA evaluation"""
        total = len(results)
        exact_matches = 0
        f1_scores = []
        precision_scores = []
        recall_scores = []
        llm_correctness_scores = []
        
        for result in results:
            # Compute exact match
            em = self.compute_exact(result.gold_answer, result.predicted_answer)
            exact_matches += em
            
            # Compute F1, precision, recall
            f1, precision, recall = self.compute_f1(result.gold_answer, result.predicted_answer)
            f1_scores.append(f1)
            precision_scores.append(precision)
            recall_scores.append(recall)
            
            # LLM correctness
            if result.llm_correctness is not None:
                llm_correctness_scores.append(100 if result.llm_correctness else 0.0)
        
        avg_execution_time = np.mean([r.execution_time for r in results])
        avg_tokens_used = np.mean([r.tokens_used for r in results])
        
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
    
    def run_experiment(self, data_file: str, output_dir: str = "results") -> Dict[str, Dict[str, float]]:
        """Run the complete experiment"""
        # Create model-specific output directory
        model_output_dir = os.path.join(output_dir, self.model)
        os.makedirs(model_output_dir, exist_ok=True)
        
        # Load data
        samples = self.load_hotpot_data(data_file)
        
        # Define baseline methods
        baseline_methods = {
            "baseline_direct": self.baseline_direct,
            "baseline_cot": self.baseline_cot,
        }
        
        # Define CEP methods - each returns two variants
        cep_methods = {
            # Understand category (2 prompts)
            "cep_understand_0": lambda s: self.cep_method(s, "understand", 0),  # Paraphrase
            "cep_understand_1": lambda s: self.cep_method(s, "understand", 1),  # Summarize
            # Connect category (2 prompts)
            "cep_connect_0": lambda s: self.cep_method(s, "connect", 0),  # What reminds you
            "cep_connect_1": lambda s: self.cep_method(s, "connect", 1),  # How relates
            # Query category (2 prompts)
            "cep_query_0": lambda s: self.cep_method(s, "query", 0),  # Most surprising
            "cep_query_1": lambda s: self.cep_method(s, "query", 1),  # Formulate questions
            # Application category (2 prompts)
            "cep_application_0": lambda s: self.cep_method(s, "application", 0),  # What can deduce
            "cep_application_1": lambda s: self.cep_method(s, "application", 1),  # Questions answered
            # Comprehensive category (1 prompt)
            "cep_comprehensive_0": lambda s: self.cep_method(s, "comprehensive", 0),  # Combined approach
        }
        
        all_results = {}
        method_metrics = {}
        
        # Run baseline methods
        for method_name, method_func in baseline_methods.items():
            logger.info(f"Running {method_name}...")
            results = []
            
            for sample in tqdm(samples, desc=method_name):
                try:
                    result = method_func(sample)
                    results.append(result)
                except Exception as e:
                    logger.error(f"Error processing sample {sample._id} with {method_name}: {e}")
                    continue
            
            # Calculate metrics for this method
            results_as_dicts = [asdict(r) for r in results]
            metrics = self.evaluator.calculate_metrics(results_as_dicts)
            method_metrics[method_name] = metrics
            
            # Save detailed results for this method
            results_file = os.path.join(model_output_dir, f"{method_name}_results.json")
            with open(results_file, 'w') as f:
                json.dump(results_as_dicts, f, indent=2)
            
            all_results[method_name] = results
            
            # Print progress for this method
            llm_score = metrics.get('llm_correctness', 0)
            logger.info(f"{method_name} - EM: {metrics['exact_match']:.3f}, F1: {metrics['f1_score']:.3f}, LLM: {llm_score:.3f}")
        
        # Run CEP methods - treat each variant as separate method
        for method_name, method_func in cep_methods.items():
            logger.info(f"Running {method_name}...")
            augmentation_results = []
            history_results = []
            
            for sample in tqdm(samples, desc=method_name):
                try:
                    augmentation_result, history_result = method_func(sample)
                    augmentation_results.append(augmentation_result)
                    history_results.append(history_result)
                except Exception as e:
                    logger.error(f"Error processing sample {sample._id} with {method_name}: {e}")
                    continue
            
            # Calculate metrics for augmentation variant
            aug_results_as_dicts = [asdict(r) for r in augmentation_results]
            augmentation_metrics = self.evaluator.calculate_metrics(aug_results_as_dicts)
            augmentation_method_name = f"cep_augmentation_{method_name.split('_', 1)[1]}"
            method_metrics[augmentation_method_name] = augmentation_metrics
            
            # Save detailed results for augmentation variant
            augmentation_results_file = os.path.join(model_output_dir, f"{augmentation_method_name}_results.json")
            with open(augmentation_results_file, 'w') as f:
                json.dump(aug_results_as_dicts, f, indent=2)
            
            all_results[augmentation_method_name] = augmentation_results
            
            # Calculate metrics for history variant
            hist_results_as_dicts = [asdict(r) for r in history_results]
            history_metrics = self.evaluator.calculate_metrics(hist_results_as_dicts)
            history_method_name = f"cep_history_{method_name.split('_', 1)[1]}"
            method_metrics[history_method_name] = history_metrics
            
            # Save detailed results for history variant
            history_results_file = os.path.join(model_output_dir, f"{history_method_name}_results.json")
            with open(history_results_file, 'w') as f:
                json.dump(hist_results_as_dicts, f, indent=2)
            
            all_results[history_method_name] = history_results
            
            # Print progress for both variants
            augmentation_llm = augmentation_metrics.get('llm_correctness', 0)
            history_llm = history_metrics.get('llm_correctness', 0)
            logger.info(f"{augmentation_method_name} - EM: {augmentation_metrics['exact_match']:.3f}, F1: {augmentation_metrics['f1_score']:.3f}, LLM: {augmentation_llm:.3f}")
            logger.info(f"{history_method_name} - EM: {history_metrics['exact_match']:.3f}, F1: {history_metrics['f1_score']:.3f}, LLM: {history_llm:.3f}")
            
            # Save updated summary metrics after each CEP method
            summary_file = os.path.join(model_output_dir, "experiment_summary.json")
            with open(summary_file, 'w') as f:
                json.dump(method_metrics, f, indent=2)
        
        # Print final summary
        print("\n" + "="*100)
        print(f"EXPERIMENT SUMMARY - {self.model.upper()}")
        print("="*100)
        print(f"{'Method':<50} {'EM':<8} {'F1':<8} {'Precision':<10} {'Recall':<8} {'LLM':<8} {'Time(s)':<10} {'Tokens':<10}")
        print("-"*100)
        
        for method_name, metrics in method_metrics.items():
            llm_score = metrics.get('llm_correctness', 0)
            print(f"{method_name:<50} {metrics['exact_match']:<8.3f} {metrics['f1_score']:<8.3f} "
                  f"{metrics.get('precision', 0):<10.3f} {metrics.get('recall', 0):<8.3f} {llm_score:<8.3f} "
                  f"{metrics['avg_execution_time']:<10.2f} {metrics['avg_tokens_used']:<10.0f}")
        
        return method_metrics

def run_multi_model_experiment(data_file: str, models: List[str], max_samples: int = 50, base_output_dir: str = "results"):
    """Run experiments across multiple models"""
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        print("Error: OPENAI_API_KEY not found in environment variables.")
        print("Please set your OpenAI API key in a .env file:")
        print("echo 'OPENAI_API_KEY=your-api-key-here' > .env")
        return
    
    all_model_results = {}
    
    for model in models:
        print(f"\n{'='*80}")
        print(f"STARTING EXPERIMENT WITH {model.upper()}")
        print(f"{'='*80}")
        
        # Create experiment instance for this model
        experiment = CEPHotPotExperiment(
            openai_api_key=openai_api_key,
            model=model,
            max_samples=max_samples
        )
        
        # Run experiment for this model
        model_results = experiment.run_experiment(data_file, base_output_dir)
        all_model_results[model] = model_results
        
        print(f"\nCompleted experiment for {model}")
    
    print(f"\nMulti-model experiment completed!")
    print(f"Results saved to {base_output_dir}/")
    
    return all_model_results

def main():
    parser = argparse.ArgumentParser(description="CEP Experiment on HotPotQA")
    parser.add_argument("--data_file", type=str, default="data/hotpot_dev_distractor_v1.json",
                       help="Path to HotPotQA JSON file")
    parser.add_argument("--model", type=str, default="all",
                       help="OpenAI model to use (or 'all' for multi-model experiment)")
    parser.add_argument("--max_samples", type=int, default=50,
                       help="Maximum number of samples to test")
    parser.add_argument("--output_dir", type=str, default="results",
                       help="Output directory for results")
    
    args = parser.parse_args()
    
    # Get OpenAI API key from environment variable
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        print("Error: OPENAI_API_KEY not found in environment variables.")
        print("Please set your OpenAI API key in a .env file:")
        print("echo 'OPENAI_API_KEY=your-api-key-here' > .env")
        print("Or set it as an environment variable:")
        print("export OPENAI_API_KEY='your-api-key-here'")
        return
    
    # Handle multi-model experiment
    if args.model.lower() == "all":
        models = ["gpt-3.5-turbo", "gpt-4o", "gpt-4o-mini"]
        print(f"Running multi-model experiment with: {', '.join(models)}")
        all_results = run_multi_model_experiment(
            data_file=args.data_file,
            models=models,
            max_samples=args.max_samples,
            base_output_dir=args.output_dir
        )
        print(f"\nMulti-model experiment completed!")
        print(f"Results saved to {args.output_dir}/")
        print("Check experiment_summary.json for detailed metrics")
    else:
        # Single model experiment
        experiment = CEPHotPotExperiment(
            openai_api_key=openai_api_key,
            model=args.model,
            max_samples=args.max_samples
        )
        
        # Run experiment
        metrics = experiment.run_experiment(args.data_file, args.output_dir)
        
        print(f"\nResults saved to {args.output_dir}/{args.model}/")
        print("Check experiment_summary.json for detailed metrics")

if __name__ == "__main__":
    main() 