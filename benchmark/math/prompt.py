import re
from typing import Callable, Dict, List

from numpy import floor
from shared.evaluator import get_step_count
from shared.experiment import CallBuilder
from shared.llm_client import LLMClient
from shared.model import CallResp, Problem
from shared.run import Call


def extract_answer(pred_str, use_last_number=True):
    pred_str = pred_str.replace("\u043a\u0438", "")
    
    pred = ""
    
    if "boxed" in pred_str:
        ans = pred_str.split("boxed")[-1]
        if len(ans) == 0:
            return ""
        elif ans[0] == "{":
            stack = 1
            a = ""
            for c in ans[1:]:
                if c == "{":
                    stack += 1
                    a += c
                elif c == "}":
                    stack -= 1
                    if stack == 0:
                        break
                    a += c
                else:
                    a += c
        else:
            a = ans.split("$")[0].strip()
        pred = a

    # multiple line
    # pred = pred.split("\n")[0]
    pred = re.sub(r"\n\s*", "", pred)
    if pred != "" and pred[0] == ":":
        pred = pred[1:]
    if pred != "" and pred[-1] == ".":
        pred = pred[:-1]
    if pred != "" and pred[-1] == "/":
        pred = pred[:-1]
    # pred = strip_string(pred)
    return pred

def build_call(model: str, template: Callable[[str], str]) -> Call:
    def call(problem: Problem) -> CallResp:
        print(f"Calling {model} with prompt: {template(problem)}")
        llm_client = LLMClient(model)
        prompt = template(problem)
        messages = [
            {"role": "user", "content": prompt}
        ]
        full_response, tokens_used, reasoning_tokens, execution_time = llm_client.call_with_history(messages, model=model)
        predicted_answer = extract_answer(full_response)
        return CallResp(
            predicted_answer=predicted_answer,
            execution_time=execution_time,
            tokens_used=tokens_used,
            reasoning_tokens=reasoning_tokens,
            chat_history=messages + [{"role": "assistant", "content": full_response}]
        )
    return call

base_cot_prompt = "Please solve this math problem step by step. Put your final answer within boxed{{}}."

def build_template(decorators: List[Callable[[str, str], str]]) -> Callable[[Problem], str]:
    def template(problem: Problem):
        prompt = base_cot_prompt
        for decorator in decorators:
            prompt = decorator(prompt, problem.problem_id) 
        return f"{prompt}\n\nQuestion: {problem.question}"
    return template

with_concise_answer = lambda prompt, problem_id: f"{prompt}\n\nProvide your reasoning in concise, manner, avoid unnecessary details."

with_point_form_answer = lambda prompt, problem_id: f"{prompt}\n\nProvide your reasoning in point form, avoiding unnecessary details."

with_headline_concise = lambda prompt, problem_id: f"{prompt}\n\nReason through this problem as if each step is a headline. Be direct and impactful."

with_telegram_persona = lambda prompt, problem_id: f"{prompt}\n\nTransmit your reasoning as if you were sending a telegram. Omit all unnecessary words."

with_critical_path = lambda prompt, problem_id: f"{prompt}\n\n Identify the single 'critical path' of logical deductions required to solve this problem. Do not explore any side-paths or dead ends. List only the steps on this critical path." 

with_efficient_language = lambda prompt, problem_id: f"""{prompt}\n\nFirst, create efficient notation for efficient representation of procedure of solving the problem. Next, solve the problem using efficient notation you created."""

with_efficient_symbol = lambda prompt, problem_id: f"{prompt}\n\nFirst, define efficient language of symbols and annotations for the problem as if you were writing a computer program. Then, solve the problem using the symbols you defined."

with_formalize = lambda prompt, problem_id: f"{prompt}\n\nFirst, formalize the problem by defining all variables, constraints, and the goal. Then, solve using only the formal structure."

with_token_budget = lambda budget: lambda prompt, problem_id: f"{prompt}\n\nPlease solve the problem with at most {budget} tokens."
def with_token_ratio_budget(ratio):
    def decorator(prompt, problem_id):
        resp = base_call_cache[problem_id]
        budget = floor(resp.tokens_used * ratio)
        return with_token_budget(budget)(prompt, problem_id)
    return decorator

with_step_limit = lambda step_limit: lambda prompt, problem_id: f"{prompt}\n\nPlease solve the problem with at most {step_limit} steps."
def with_step_ratio_limit(ratio):
    def decorator(prompt, problem_id):
        resp = base_call_cache[problem_id]
        step_count = get_step_count(resp.chat_history)
        step_limit = floor(step_count * ratio)
        return with_step_limit(step_limit)(prompt, problem_id)
    return decorator

base_call_cache = {}

def base_wrapper(base_call:Call):
    def wrapped_call(problem:Problem):
        call_resp = base_call(problem)
        base_call_cache[problem.problem_id] = call_resp
        return call_resp
    return wrapped_call

class MathCallBuilder(CallBuilder):
    """Call builder for math problems"""
    
    def build_calls(self, model: str, domain: str, with_cot: bool) -> Dict[str, Call]:
        """Build calls for math problems"""
        calls = {}
        prompts = {
            "base": [],
            #"with_concise_answer": [with_concise_answer],
            #"with_point_form_answer": [with_point_form_answer],
            #"with_headline_concise": [with_headline_concise],
            #"with_telegram_persona": [with_telegram_persona],
            #"with_critical_path": [with_critical_path],
            #"with_efficient_language": [with_efficient_language],
            #"with_formalize": [with_formalize],
            #"with_token_budget_50": [with_token_budget(50)],
            #"with_token_budget_100": [with_token_budget(100)],
            #"with_token_budget_200": [with_token_budget(200)],
            #"with_token_budget_400": [with_token_budget(400)],
            #"with_token_budget_600": [with_token_budget(600)],
            "with_token_ratio_budget_0.25": [with_token_ratio_budget(0.25)],
            "with_token_ratio_budget_0.5": [with_token_ratio_budget(0.5)],
            "with_token_ratio_budget_1.0": [with_token_ratio_budget(1.0)],
            "with_token_ratio_budget_2.0": [with_token_ratio_budget(2.0)],
            "with_token_ratio_budget_4.0": [with_token_ratio_budget(4.0)],
            #"with_efficient_symbol": [with_efficient_symbol],
            #"with_step_limit_5": [with_step_limit(5)],
            #"with_step_limit_15": [with_step_limit(15)],
            #"with_step_limit_40": [with_step_limit(40)],
            "with_step_ratio_limit_0.25": [with_step_ratio_limit(0.25)],
            "with_step_ratio_limit_0.5": [with_step_ratio_limit(0.5)],
            "with_step_ratio_limit_1.0": [with_step_ratio_limit(1.0)],
            "with_step_ratio_limit_2.0": [with_step_ratio_limit(2.0)],
            "with_step_ratio_limit_4.0": [with_step_ratio_limit(4.0)],
        }

        for prompt_name, decorators in prompts.items():
            template = build_template(decorators)
            call = build_call(model, template)
            if prompt_name == "base":
                call = base_wrapper(call)
            calls[prompt_name] = call
        return calls

