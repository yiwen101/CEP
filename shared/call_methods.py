from typing import Dict, List, Tuple
from shared.llm_client import LLMClient
from shared.model import CallResp, ElaborationData, Problem
from shared.run import Call


# Global cache for elaborations to ensure cep_augmentation and cep_history use the same elaboration
_elaboration_cache: Dict[str, ElaborationData] = {}


def create_baseline_direct_call(model: str) -> Call:
    """Create a baseline direct call function"""
    llm_client = LLMClient(model)
    
    def baseline_direct_call(problem: Problem) -> CallResp:
        """Baseline: Direct question answering without elaboration"""
        messages = [
            {"role": "user", "content": f"""Answer the following question with reference to the context:
Question: {problem.question}

Context:
{problem.context}

Please think step by step and then provide your answer:"""}
        ]
        
        predicted_answer, tokens_used, execution_time = llm_client.call_with_history(messages)
        
        # Create full chat history including the response
        full_history = messages + [{"role": "assistant", "content": predicted_answer}]
        
        return CallResp(
            predicted_answer=predicted_answer,
            execution_time=execution_time,
            tokens_used=tokens_used,
            chat_history=full_history
        )
    
    return baseline_direct_call


def create_baseline_cot_call(model: str) -> Call:
    """Create a baseline Chain-of-Thought call function"""
    llm_client = LLMClient(model)
    
    def baseline_cot_call(problem: Problem) -> CallResp:
        """Baseline: Chain-of-Thought reasoning"""
        messages = [
            {"role": "user", "content": f"""Answer the following question with reference to the context:
Question: {problem.question}

Context:
{problem.context}

Please think step by step and then provide your answer:"""}
        ]
        
        predicted_answer, tokens_used, execution_time = llm_client.call_with_history(messages)
        
        # Create full chat history including the response
        full_history = messages + [{"role": "assistant", "content": predicted_answer}]
        
        return CallResp(
            predicted_answer=predicted_answer,
            execution_time=execution_time,
            tokens_used=tokens_used,
            chat_history=full_history
        )
    
    return baseline_cot_call


def build_elaboration_in_turns(model: str, context: str, ceps: List[str]) -> ElaborationData:
    """Build elaboration by asking CEPs one by one"""
    if len(ceps) == 0:
        raise ValueError("CEPs list is empty")
    tokens_used = 0
    execution_time = 0
    elaborations = []
    
    llm_client = LLMClient(model)
    messages = [
        {"role": "user", "content": f"""Context:
{context}

Question: {ceps[0]}

provide your answer:"""}
    ]
    elaboration, tokens, time = llm_client.call_with_history(messages)
    elaborations.append(elaboration)
    tokens_used += tokens
    execution_time += time
    messages.append({"role": "assistant", "content": elaboration})

    for cep in ceps[1:]:
        messages.append({"role": "user", "content": f"""{cep}"""})
        elaboration, tokens, time = llm_client.call_with_history(messages)
        elaborations.append(elaboration)
        tokens_used += tokens
        execution_time += time
        messages.append({"role": "assistant", "content": elaboration})
    
    return ElaborationData(
        elaboration="\n".join(elaborations), 
        elaboration_history=messages, 
        tokens_used=tokens_used, 
        execution_time=execution_time
    )


def build_elaboration_in_one_go(model: str, context: str, ceps: List[str]) -> ElaborationData:
    """Build elaboration by asking all CEPs at once"""
    return build_elaboration_in_turns(model, context, ["\n".join(ceps)])


def _get_or_create_elaboration(problem: Problem, model: str, cep_prompts: List[str]) -> ElaborationData:
    """Get elaboration from cache or create new one using build_elaboration_in_one_go"""
    # Create a cache key based on problem_id, model, and the CEP prompts
    cache_key = f"{problem.problem_id}_{model}_{hash(tuple(cep_prompts))}"
    
    if cache_key not in _elaboration_cache:
        # Use build_elaboration_in_one_go as requested
        elaboration_data = build_elaboration_in_one_go(model, problem.context, cep_prompts)
        _elaboration_cache[cache_key] = elaboration_data
    
    return _elaboration_cache[cache_key]


def create_cep_augmentation_call(model: str, cep_prompts: List[str], with_cot: bool) -> Call:
    """Create a CEP augmentation call function"""
    llm_client = LLMClient(model)
    
    def cep_augmentation_call(problem: Problem) -> CallResp:
        # Get or create elaboration using the same key as cep_history
        elaboration_data = _get_or_create_elaboration(problem, model, cep_prompts)
        
        elaboration = elaboration_data.elaboration
        tokens_1 = elaboration_data.tokens_used
        time_1 = elaboration_data.execution_time
        
        # Step 2: Question answering with elaborated context
        augmentation_query = f"""Answer the following question with reference to the context:
Question: {problem.question}

Context:
{problem.context}
{elaboration}

{"Please think step by step and then "if with_cot else ""}provide your answer:"""

        augmentation_messages = [
            {"role": "user", "content": augmentation_query}
        ]
        
        predicted_answer, tokens_2, time_2 = llm_client.call_with_history(augmentation_messages)
        
        # Combine chat history including all responses
        full_history = (
            elaboration_data.elaboration_history + 
            augmentation_messages + 
            [{"role": "assistant", "content": predicted_answer}]
        )
        
        return CallResp(
            predicted_answer=predicted_answer,
            execution_time=time_1 + time_2,
            tokens_used=tokens_1 + tokens_2,
            chat_history=full_history,
            elaboration=elaboration
        )
    
    return cep_augmentation_call


def create_cep_history_call(model: str, cep_prompts: List[str], with_cot: bool) -> Call:
    """Create a CEP history call function"""
    llm_client = LLMClient(model)
    
    def cep_history_call(problem: Problem) -> CallResp:
        """CEP history: context elaboration then question answering"""
        # Get or create elaboration using the same key as cep_augmentation
        elaboration_data = _get_or_create_elaboration(problem, model, cep_prompts)
        
        elaboration = elaboration_data.elaboration
        tokens_1 = elaboration_data.tokens_used
        time_1 = elaboration_data.execution_time
        elaboration_history = elaboration_data.elaboration_history
        
        history_messages = elaboration_history.copy()
        history_messages.extend([
            {"role": "user", "content": f"""Answer the following question with reference to the context:
Question: {problem.question}

Context:
{problem.context}

{"Please think step by step and then "if with_cot else ""}provide your answer:"""}])

        predicted_answer, tokens_2, time_2 = llm_client.call_with_history(history_messages)

        full_history = history_messages + [{"role": "assistant", "content": predicted_answer}]

        return CallResp(
            predicted_answer=predicted_answer,
            execution_time=time_1 + time_2,
            tokens_used=tokens_1 + tokens_2,
            chat_history=full_history,
            elaboration=elaboration
        )
    
    return cep_history_call


def clear_elaboration_cache():
    """Clear the elaboration cache"""
    global _elaboration_cache
    _elaboration_cache.clear()


def get_cache_info() -> Dict[str, int]:
    """Get information about the elaboration cache"""
    return {
        "cache_size": len(_elaboration_cache),
        "cache_keys": list(_elaboration_cache.keys())
    }