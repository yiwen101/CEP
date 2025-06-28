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
from dotenv import load_dotenv
load_dotenv()

logger = logging.getLogger(__name__)
API_KEY = os.getenv("DEEPSEEK_API_KEY")

class DeepSeekClient:
    """Shared LLM client for making OpenAI API calls with retry logic"""
    
    def __init__(self, model: str = "gpt-3.5-turbo"):
        self.client = OpenAI(api_key=API_KEY, base_url="https://api.deepseek.com")
        self.model = 'deepseek-reasoner'
    
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
                reasoning_content = response.choices[0].message.reasoning_content
                content = response.choices[0].message.content
                tokens_used = response.usage.completion_tokens
                reason_token = response.usage.completion_tokens_details.reasoning_tokens
                
                return reasoning_content, content, tokens_used, reason_token, execution_time
                
            except Exception as e:
                if attempt == max_retries - 1:  # Last attempt
                    logger.error(f"OpenAI API call failed after {max_retries} attempts: {e}")
                    return f"Error: {e}", 0, 0.0
                
                # Calculate delay with exponential backoff and jitter
                delay = base_delay * (2 ** attempt) + random.uniform(0, 1)
                logger.warning(f"API call failed (attempt {attempt + 1}/{max_retries}), retrying in {delay:.2f}s: {e}")
                time.sleep(delay)