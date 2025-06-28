# load data/math/test.jsonl
import json
import random
from shared.deepseek import DeepSeekClient

client = DeepSeekClient()

with open("data/math/test.jsonl", "r") as f:
    objs = [json.loads(line) for line in f]
    # get one random line
    obj = random.choice(objs)
    question = obj["problem"]
    prompt = f"Please solve this math problem. Put your final answer within boxed{{}}.\n\nQuestion: {question}"
    answer = obj["answer"]
    print(prompt)
    print(answer)
    reasoning_content, response, tokens_used, reasoning_tokens, execution_time = client.call_with_history([{"role": "user", "content": prompt}], "deepseek-reasoner")
    print(reasoning_content)
    print(response)
    print(tokens_used)
    print(reasoning_tokens)
    print(execution_time)
