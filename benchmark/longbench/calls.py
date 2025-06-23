"""
LongBenchV2 call methods for CEP experiments.
"""

import re
from typing import Dict
from shared import Call, CallResp, LLMClient, CEPPrompts, CallBuilder, model
from shared.call_methods import create_baseline_direct_call, create_baseline_cot_call, create_cep_augmentation_call, create_cep_history_call


class LongBenchCallBuilder(CallBuilder):
    """CallBuilder for LongBenchV2 experiments"""
    
    def build_calls(self, model: str, domain: str, with_cot: bool) -> Dict[str, Call]:
        """Build a map of method names to call functions for a given model and domain"""
        calls = {}
        
        # Baseline methods
        calls["baseline_direct"] = self.build_wrapper_calls(create_baseline_direct_call(model))
        calls["baseline_cot"] = self.build_wrapper_calls(create_baseline_cot_call(model))
        
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
                calls[method_name] = self.build_wrapper_calls(create_cep_augmentation_call(model, cep_prompts_array, with_cot))
                
                # Add history variant
                method_name = f"cep_history_{category}"
                if num_prompts > 1:
                    method_name += f"_{prompt_index}"
                calls[method_name] = self.build_wrapper_calls(create_cep_history_call(model, cep_prompts_array, with_cot))
        
        return calls
    
    def build_wrapper_calls(self, call: Call) -> Call:
        """Build a wrapper call that extracts the answer from the response"""
        def wrapper_call(problem: model.Problem) -> CallResp:
            resp = call(problem)
            print(resp.predicted_answer)
            resp.predicted_answer = self.extract_answer(resp.predicted_answer)
            return resp
        return wrapper_call
    
    def extract_answer(self, response: str) -> str:
        """Extract the answer choice from the model response"""
        response = response.replace('*', '')
        match = re.search(r'correct answer is \(([A-Da-z])\)', response)
        if match:
            return match.group(1)
        else:
            match = re.search(r'correct answer is ([A-Da-z])', response)
            if match:
                return match.group(1)
            else:
                return "wrong answer"