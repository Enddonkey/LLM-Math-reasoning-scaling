# DDA5001 Final Project - Part III: Test-Time Scaling and Reasoning

This project corresponds to Part III of the DDA5001 Final Project. It focuses on exploring the effects of test-time scaling techniques on the mathematical reasoning capabilities of Large Language Models (LLMs).

## Project Overview

The primary goal is to analyze how different sampling strategies, specifically temperature variation and multi-sample generation (`pass@k`), impact the performance of math-specialist LLMs on various benchmarks.

-   **Models**:
    -   Base: `Qwen/Qwen2.5-Math-1.5B`
    -   Instruct-tuned: `Qwen/Qwen2.5-Math-1.5B-Instruct`
-   **Datasets**:
    -   `Math500`
    -   `AMC23`
    -   `AIME25`
-   **Key Concepts**:
    -   **Test-Time Scaling**: Improving model performance at inference time without additional training.
    -   **Temperature Sampling**: Adjusting the randomness of the model's output.
    -   **pass@k**: A metric to evaluate the probability of finding a correct answer within `k` generated samples.
    -   **Majority Vote (`maj@1`)**: An aggregation strategy where the most frequent answer among multiple samples is chosen.

## Project Structure

```
.
├── dda5001-part3.ipynb      # Main Jupyter notebook with the complete workflow.
├── project manual-part III.pdf # Project description and instructions.
├── README.md                # This file.
├── requirements.txt         # Full list of dependencies for conda.
├── src/                     # Source code directory.
│   ├── inference.py         # Script for running model inference and generation.
│   ├── evaluate.py          # Script for scoring generations and calculating metrics.
│   ├── analysis.py          # Script for aggregating results and plotting.
│   └── verifier/            # Helper module for answer verification.
└── outputs/                 # Directory for all generated files.
    ├── *.jsonl              # Raw and evaluated generation outputs.
    ├── summary.csv          # Aggregated results in CSV format.
    ├── all_results.json     # Aggregated results in JSON format.
    └── plots/               # Directory for generated plots.
```

## Setup and Installation

It is recommended to set up the environment using `conda` or `pip`. The key dependencies are listed below.

```bash
# Key dependencies
pip install torch
pip install vllm
pip install transformers datasets
pip install pandas seaborn matplotlib
pip install tqdm pylatexenc 'math-verify[antlr4_9_3]'
```
For a complete list of packages, refer to `requirements.txt`.

## Workflow

The entire experimental process is orchestrated in `dda5001-part3.ipynb` and can be broken down into three main stages.

### 1. Generation (Inference)

Use the `src/inference.py` script to generate solutions for the problems in each dataset. The script is run for each combination of model, dataset, and temperature.

**Example Command:**
```bash
python src/inference.py \
  --model "Qwen/Qwen2.5-Math-1.5B" \
  --dataset "math" \
  --dp-size 2 \
  --batch-size 16 \
  --rollout-n 16 \
  --temperature 0.6 \
  --top-p 0.9 \
  --output_file outputs/math500_base_t0.6.jsonl
```
-   `--rollout-n` should be `16` for `Math500` and `64` for `AMC23` and `AIME25`.
-   `--temperature` is varied across `0.6`, `1.0`, and `1.2`.

### 2. Evaluation

After generation, use the `src/evaluate.py` script to score the generated answers. This script processes the output from the inference step, computes correctness for each sample, and saves an `_eval.jsonl` file.

**Example Command:**
```bash
python src/evaluate.py \
  --input_file outputs/math500_base_t0.6.jsonl \
  --output_file outputs/math500_base_t0.6_eval.jsonl
```

### 3. Analysis

Finally, run the `src/analysis.py` script to aggregate the metrics from all `_eval.jsonl` files. This script generates summary files (`summary.csv`, `all_results.json`) and a series of plots in the `outputs/plots/` directory, which are used for the final analysis.

**Example Command:**
```bash
python src/analysis.py --input_dir outputs --output_dir outputs
```

This will produce:
-   Heatmaps for `pass@k` and `maj@1` scores.
-   Line plots showing `pass@k` vs. temperature.
-   Bar plots comparing models and dataset performance.
