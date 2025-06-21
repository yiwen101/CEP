#!/usr/bin/env python3
"""
Example script demonstrating CEP experiment on HotPotQA
This script runs a small experiment with a few samples to demonstrate the methods
"""

import os
import json
from dotenv import load_dotenv
from cep_hotpot_experiment import CEPHotPotExperiment

# Load environment variables from .env file
load_dotenv()

def create_sample_data():
    """Create a small sample dataset for demonstration"""
    sample_data = [
        {
            "_id": "sample_1",
            "question": "Which country has a larger population, China or India?",
            "answer": "China",
            "supporting_facts": [["China", 0], ["India", 0]],
            "context": [
                ["China", ["China is the most populous country in the world with over 1.4 billion people.", "It is located in East Asia and has the world's second-largest economy."]],
                ["India", ["India is the second most populous country with over 1.3 billion people.", "It is located in South Asia and is the world's largest democracy."]]
            ],
            "type": "comparison",
            "level": "easy"
        },
        {
            "_id": "sample_2", 
            "question": "What is the capital of the country where the Eiffel Tower is located?",
            "answer": "Paris",
            "supporting_facts": [["Eiffel Tower", 0], ["France", 0]],
            "context": [
                ["Eiffel Tower", ["The Eiffel Tower is a famous landmark located in Paris, France.", "It was completed in 1889 and stands 324 meters tall."]],
                ["France", ["France is a country in Western Europe with Paris as its capital city.", "It is known for its culture, cuisine, and art."]]
            ],
            "type": "bridge",
            "level": "easy"
        }
    ]
    
    # Save sample data
    os.makedirs("data", exist_ok=True)
    with open("data/sample_hotpot.json", "w") as f:
        json.dump(sample_data, f, indent=2)
    
    return "data/sample_hotpot.json"

def main():
    """Run a small example experiment"""
    
    # Check if OpenAI API key is set
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Error: OPENAI_API_KEY not found in environment variables.")
        print("Please set your OpenAI API key in a .env file:")
        print("echo 'OPENAI_API_KEY=your-api-key-here' > .env")
        print("Or set it as an environment variable:")
        print("export OPENAI_API_KEY='your-api-key-here'")
        return
    
    # Create sample data
    data_file = create_sample_data()
    
    # Initialize experiment with small sample size
    experiment = CEPHotPotExperiment(
        openai_api_key=api_key,
        model="gpt-4",  # or "gpt-3.5-turbo" for faster/cheaper testing
        max_samples=2
    )
    
    print("Running CEP experiment on sample HotPotQA data...")
    print("This will test the effectiveness of Context Elaboration Prompts")
    print("=" * 60)
    
    # Run experiment
    metrics = experiment.run_experiment(data_file, "example_results")
    
    print("\n" + "=" * 60)
    print("EXAMPLE EXPERIMENT COMPLETED")
    print("=" * 60)
    print("Check the 'example_results' directory for detailed outputs:")
    print("- experiment_summary.json: Overall metrics")
    print("- {method}_results.json: Detailed results for each method")
    
    # Show a sample result
    print("\nSample result from ICE with intention (comprehensive):")
    try:
        with open("example_results/ice_with_intention_comprehensive_results.json", "r") as f:
            results = json.load(f)
            if results:
                sample = results[0]
                print(f"Question: {sample['question']}")
                print(f"Gold Answer: {sample['gold_answer']}")
                print(f"Predicted Answer: {sample['predicted_answer']}")
                print(f"Execution Time: {sample['execution_time']:.2f}s")
                print(f"Tokens Used: {sample['tokens_used']}")
    except FileNotFoundError:
        print("Results file not found - experiment may have failed")

if __name__ == "__main__":
    main() 