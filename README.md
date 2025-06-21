# Context Elaboration Prompt (CEP) Experiment on HotPotQA

This repository implements experiments to test the effectiveness of Context Elaboration Prompts (CEPs) for enhancing LLM reasoning on multi-hop question answering tasks, specifically using the HotPotQA dataset.

## Overview

The research investigates whether structured context elaboration can improve LLM performance on complex reasoning tasks by:
1. **Two-Stage Context Elaboration**: First elaborating context without query awareness, then answering queries using elaborated context
2. **Context Augmentation**: Using elaborated context alongside original context
3. **Baseline Comparison**: Comparing against direct QA and Chain-of-Thought methods

## Features

- **Multiple CEP Types**: Understand, Connect, Query, Application, and Comprehensive elaboration prompts
- **Structured Output Evaluation**: Uses Pydantic models for reliable LLM-based correctness evaluation
- **Exponential Backoff**: Robust API handling with retry logic and jitter
- **Comprehensive Metrics**: Exact Match, F1 Score, LLM Correctness, Confidence Analysis
- **Detailed Analysis**: Performance comparison, efficiency analysis, and correlation studies

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd CEP
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up your OpenAI API key:
```bash
echo 'OPENAI_API_KEY=your-api-key-here' > .env
```

## Usage

### 1. Download HotPotQA Data

```bash
python download_hotpot_data.py
```

### 2. Run a Small Demo

```bash
python example_run.py
```

### 3. Run Full Experiment

```bash
# Single model experiment
python cep_hotpot_experiment.py --data_file hotpot_qa_data.json --max_samples 50

# Multi-model experiment (gpt-3.5-turbo, gpt-4o, gpt-4o-mini)
python cep_hotpot_experiment.py --data_file hotpot_qa_data.json --max_samples 50 --model all

# Or use the dedicated multi-model script
python run_multi_model_experiment.py
```

### 4. Analyze Results

```bash
python analyze_results.py --results_dir results
```

## Methods Tested

### Baseline Methods
- **Direct**: Standard question answering without elaboration
- **Chain-of-Thought (CoT)**: Step-by-step reasoning approach

### CEP Methods
- **CEP Elaboration**: Turn-based approach using only elaborated context
  - `cep_understand_0`: "Paraphrase the provided information in your own words"
  - `cep_understand_1`: "Summarize the given text in a clear and concise manner"
  - `cep_connect_0`: "What does this information remind you of? Briefly explain the connection."
  - `cep_connect_1`: "How does this information relate to other facts or concepts you know?"
  - `cep_query_0`: "What do you find to be the most surprising or interesting piece of information?"
  - `cep_query_1`: "Formulate two insightful questions that are raised by the text"
  - `cep_application_0`: "What can you deduce from the given information?"
  - `cep_application_1`: "Formulate two insightful questions that are answered by the information given"
  - `cep_comprehensive_0`: Combined approach with all 4 types of prompts

### CEP Categories
- **Understand**: "Paraphrase the provided information in your own words"
- **Connect**: "What does this information remind you of? Briefly explain the connection"
- **Query**: "What do you find to be the most surprising or interesting piece of information?"
- **Application**: "What can you deduce from the given information?"
- **Comprehensive**: Multi-faceted elaboration covering entities, relationships, and structure

## Multi-Model Experiments

The system supports running experiments across multiple OpenAI models to compare performance:

### Supported Models
- **gpt-3.5-turbo**: Fast and cost-effective baseline
- **gpt-4o**: High-performance model for detailed analysis
- **gpt-4o-mini**: Balanced performance and cost

### Multi-Model Features
- **Organized Results**: Each model's results are saved in separate directories
- **Incremental Saving**: Results are saved after each method completes
- **Simple Execution**: Use `--model all` to run all models

### Directory Structure
```
results/
├── gpt-3.5-turbo/
│   ├── experiment_summary.json
│   ├── baseline_direct_results.json
│   ├── baseline_cot_results.json
│   └── ...
├── gpt-4o/
│   ├── experiment_summary.json
│   └── ...
└── gpt-4o-mini/
    ├── experiment_summary.json
    └── ...
```

## Evaluation Metrics

### Primary Metrics
- **Exact Match (EM)**: String-level accuracy comparison
- **F1 Score**: Token-level overlap between predicted and gold answers
- **LLM Correctness**: AI-powered evaluation using simple text output

### Secondary Metrics
- **Efficiency Metrics**: Performance per token and per second
- **Correlation Analysis**: Relationships between different metrics

### LLM-Based Evaluation

The system uses an LLM to evaluate answer correctness with a simple text-based approach:

- **Simple Output**: LLM outputs only "correct" or "wrong" 
- **Semantic Understanding**: Considers semantic equivalence (e.g., "USA" vs "United States")
- **Fallback Mechanism**: Uses exact match if LLM evaluation fails
- **Robust Parsing**: Handles various response formats with intelligent parsing

## Output Files

### Experiment Results
- `results/{model_name}/experiment_summary.json`: Overall metrics for all methods for a specific model
- `results/{model_name}/{method_name}_results.json`: Detailed results for each method for a specific model

### Analysis Visualizations
- `performance_comparison.png`: Bar charts comparing all metrics
- `efficiency_analysis.png`: Performance vs. cost analysis
- `metric_correlation.png`: Correlation matrix between metrics

### Reports
- `analysis_report.txt`: Comprehensive text report with rankings and insights

## API Rate Limiting

The implementation includes robust handling for OpenAI API rate limits:

- **Exponential Backoff**: Retry delays increase exponentially (1s, 2s, 4s, 8s...)
- **Jitter**: Random delay component prevents thundering herd
- **Configurable Retries**: Default 10 retries for regular calls
- **Graceful Degradation**: Fallback to exact match if LLM evaluation fails

## Example Output

```
EXPERIMENT SUMMARY - GPT-3.5-TURBO
====================================================================================================
Method                                    EM      F1      LLM     Time(s)   Tokens   
----------------------------------------------------------------------------------------------------
cep_comprehensive_0                       0.420   0.456   0.380   12.34     1250     
cep_understand_0                          0.395   0.428   0.350   11.23     1200     
cep_understand_1                          0.390   0.425   0.345   11.45     1220     
cep_connect_0                             0.385   0.418   0.340   11.67     1240     
cep_connect_1                             0.380   0.415   0.335   11.89     1260     
cep_query_0                               0.375   0.410   0.330   12.11     1280     
cep_query_1                               0.370   0.405   0.325   12.33     1300     
cep_application_0                         0.365   0.400   0.320   12.55     1320     
cep_application_1                         0.360   0.395   0.315   12.77     1340     
baseline_cot                             0.380   0.425   0.340   8.90      1100     
baseline_direct                          0.350   0.395   0.315   6.45      950      
```

## Research Context

This implementation tests the methods proposed in the research on "Enhancing LLM Reasoning via Structured Context Elaboration":

- **Core Hypothesis**: Structured elaboration improves context assimilation and reasoning
- **Germane Load Theory**: Focuses on mental resources spent on schema formation
- **Multi-Hop Reasoning**: Tests on HotPotQA's complex multi-document reasoning tasks

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use this code in your research, please cite:

```bibtex
@misc{cep_hotpot_experiment,
  title={Context Elaboration Prompt Experiments on HotPotQA},
  author={Your Name},
  year={2024},
  url={https://github.com/yourusername/cep-experiment}
}
```

## CEP Method Implementation

The experiment implements a unified **Context Elaboration Prompt (CEP)** method that returns two variants for each prompt:

### Method Variants
1. **Augmentation Variant**: Uses a modified query that includes both original and elaborated context
   - No conversation history
   - Query format: `Original Context + Elaborated Context + Question`
   
2. **History Variant**: Uses full conversation history
   - Maintains conversation history between turns
   - Query format: `Context + CEP → Elaboration → Question`

### Process Flow
1. **Elaboration Phase**: Context + CEP → Elaboration (same for both variants)
2. **Answering Phase**: 
   - **Augmentation**: Modified query with both contexts
   - **History**: Full conversation history with the question

### CEP Categories
Each category contains multiple prompts that are tested individually:

- **Understand**: "Paraphrase the provided information in your own words", "summarize the given text"
- **Connect**: "What does this information remind you of? Briefly explain the connection."
- **Query**: "What do you find to be the most surprising or interesting piece of information?", "Formulate two insightful questions that are raised by the text"
- **Application**: "What can you deduce from the given information?", "Formulate two insightful questions that are answered by the information given"
- **Comprehensive**: A combined prompt covering all categories