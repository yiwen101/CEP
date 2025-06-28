"""
Shared LLM Client for CEP Experiments
Handles OpenAI API calls with retry logic, token counting, and execution time tracking
"""

import os
import time
import random
import logging
from typing import Dict, List, Tuple, Optional
from openai import OpenAI

logger = logging.getLogger(__name__)
API_KEY = os.getenv("OPENAI_API_KEY")

class LLMClient:
    """Shared LLM client for making OpenAI API calls with retry logic"""
    
    def __init__(self, model: str = "gpt-3.5-turbo"):
        self.client = OpenAI(api_key=API_KEY)
        self.model = model
    
    def call_with_history(self, messages: List[Dict[str, str]], model: str, max_tokens: int = 2000, 
                         max_retries: int = 10, base_delay: float = 1.0) -> Tuple[str, int, float]:
        """
        Make a call to OpenAI API with conversation history and return response with token count
        
        Args:
            messages: List of message dictionaries with 'role' and 'content'
            max_tokens: Maximum tokens for response
            temperature: Sampling temperature
            model: Model to use (defaults to self.model)
            max_retries: Maximum number of retry attempts
            base_delay: Base delay for exponential backoff
            
        Returns:
            Tuple of (response_content, tokens_used, execution_time)
        """
        for attempt in range(max_retries):
            try:
                start_time = time.time()
                response = self.client.chat.completions.create(
                    model=model,
                    messages=messages,
                    max_completion_tokens=max_tokens
                )
                execution_time = time.time() - start_time
                
                content = response.choices[0].message.content
                tokens_used = response.usage.completion_tokens
                print(response.usage)
                reason_token = response.usage.completion_tokens_details.reasoning_tokens
                
                return content, tokens_used, reason_token, execution_time
                
            except Exception as e:
                if attempt == max_retries - 1:  # Last attempt
                    logger.error(f"OpenAI API call failed after {max_retries} attempts: {e}")
                    return f"Error: {e}", 0, 0.0
                
                # Calculate delay with exponential backoff and jitter
                delay = base_delay * (2 ** attempt) + random.uniform(0, 1)
                logger.warning(f"API call failed (attempt {attempt + 1}/{max_retries}), retrying in {delay:.2f}s: {e}")
                time.sleep(delay)
    

    def evaluate_mcq_correctness(self, choices: str, answer: str, predicted_answer: str) -> bool:
        """
        Evaluate correctness of a choice question
        """
        messages = [
            {"role": "user", "content": f"""
            You are a helpful assistant that evaluates the correctness of a choice question.
            Choices: {choices}
            Answer: {answer}
            Predicted Answer: {predicted_answer}
            If the predicted answer is correct, respond with "correct" else respond with "wrong". Respond with only "correct" or "wrong".
            """}
        ]
        
        response, _, _, _ = self.call_with_history(messages, max_tokens=10, temperature=0.0)
        return response.strip().lower() == "correct" 


    def evaluate_correctness(self, question: str, gold_answer: str, predicted_answer: str) -> bool:
        """
        Evaluate correctness using LLM with simple prompt
        
        Args:
            question: The question being answered
            gold_answer: The correct answer
            predicted_answer: The model's predicted answer
            
        Returns:
            True if the answer is correct, False otherwise
        """
        messages = [
            {"role": "user", "content": f"""Question: {question}
Gold Answer: {gold_answer}
Predicted Answer: {predicted_answer}

Is the predicted answer correct? Respond with only "correct" or "wrong"."""}
        ]
        
        response, _, _, _ = self.call_with_history(messages, max_tokens=10, temperature=0.0)
        return response.strip().lower() == "correct" 