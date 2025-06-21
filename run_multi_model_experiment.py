#!/usr/bin/env python3
"""
Multi-model CEP experiment runner
Runs experiments across gpt-3.5-turbo, gpt-4o, and gpt-4o-mini
"""

import os
import sys
from cep_hotpot_experiment import run_multi_model_experiment

def main():
    # Configuration
    data_file = "data/hotpot_dev_distractor_v1.json"  # Update this path as needed
    max_samples = 50  # Adjust as needed
    output_dir = "results"
    
    # Models to test
    models = ["gpt-3.5-turbo", "gpt-4o", "gpt-4o-mini"]
    
    print("="*80)
    print("MULTI-MODEL CEP EXPERIMENT")
    print("="*80)
    print(f"Data file: {data_file}")
    print(f"Models: {', '.join(models)}")
    print(f"Max samples per model: {max_samples}")
    print(f"Output directory: {output_dir}")
    print("="*80)
    
    # Check if data file exists
    if not os.path.exists(data_file):
        print(f"Error: Data file {data_file} not found!")
        print("Please download the HotPotQA data first:")
        print("python download_hotpot_data.py")
        return
    
    # Check if API key is set
    if not os.getenv("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY not found in environment variables.")
        print("Please set your OpenAI API key in a .env file:")
        print("echo 'OPENAI_API_KEY=your-api-key-here' > .env")
        return
    
    # Run the multi-model experiment
    try:
        all_results = run_multi_model_experiment(
            data_file=data_file,
            models=models,
            max_samples=max_samples,
            base_output_dir=output_dir
        )
        
        print("\n" + "="*80)
        print("EXPERIMENT COMPLETED SUCCESSFULLY!")
        print("="*80)
        print(f"Results saved to:")
        for model in models:
            print(f"  - {output_dir}/{model}/")
        print(f"  - {output_dir}/cross_model_comparison.json")
        
        print("\nTo analyze results, run:")
        print("python analyze_results.py --results_dir results")
        
    except Exception as e:
        print(f"Error during experiment: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 