"""
HotPotQA Call Functions for CEP Experiments
"""

import os
from typing import List, Dict
from dotenv import load_dotenv

from shared import Call, CallResp, LLMClient, CEPPrompts, CallBuilder
from shared.call_methods import (
    create_baseline_direct_call,
    create_baseline_cot_call,
    create_cep_augmentation_call,
    create_cep_history_call
)

# Load environment variables
load_dotenv()

class HotpotCallBuilder(CallBuilder):
    """CallBuilder for HotPotQA experiments"""
    
    def build_calls(self, model: str, domain: str, with_cot: bool) -> Dict[str, Call]:
        """Build a map of method names to call functions for a given model and domain"""
        calls = {}
        
        # Baseline methods
        calls["baseline_direct"] = create_baseline_direct_call(model)
        calls["baseline_cot"] = create_baseline_cot_call(model)
        
        # CEP methods - each category with augmentation and history variants
        cep_categories = ["understand", "connect", "query", "application", "comprehensive"]
        cep_prompts = CEPPrompts()
        
        for category in cep_categories:
            # Get number of prompts for this category
            all_ceps = cep_prompts.get_all_ceps("general")
            num_prompts = len(all_ceps.get(category, []))
            
            for prompt_index in range(num_prompts):
                # Get the specific CEP prompt
                cep_prompt = cep_prompts.get_cep_prompt(category, prompt_index, "general")
                cep_prompts_array = [cep_prompt]
                
                # Add augmentation variant
                method_name = f"cep_augmentation_{category}"
                if num_prompts > 1:
                    method_name += f"_{prompt_index}"
                calls[method_name] = create_cep_augmentation_call(model, cep_prompts_array, with_cot)
                
                # Add history variant
                method_name = f"cep_history_{category}"
                if num_prompts > 1:
                    method_name += f"_{prompt_index}"
                calls[method_name] = create_cep_history_call(model, cep_prompts_array, with_cot)
        
        return calls
