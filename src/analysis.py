import json
import argparse
from tqdm import tqdm
from pathlib import Path
from collections import defaultdict, Counter
import math
import random
import re
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

# Import the same verifier used by evaluate.py
from verifier import compute_score

# Fix random seed for tie-breaking in Majority Vote, ensuring consistency
random.seed(42)


# --- Plotting Functions (from original analysis.py) ---

def plot_pass_at_k_by_temperature(df, output_dir):
    """Plots pass@k vs. temperature for each model and dataset."""
    for model in df['model'].unique():
        for dataset in df['dataset'].unique():
            subset = df[(df['model'] == model) & (df['dataset'] == dataset)]
            if subset.empty:
                continue

            plt.figure(figsize=(10, 6))
            # Use all available pass@k columns
            pass_k_cols = sorted([col for col in subset.columns if col.startswith('pass@')], key=lambda x: int(x.split('@')[1]))
            for pass_k_col in pass_k_cols:
                sns.lineplot(data=subset, x='temperature', y=pass_k_col, marker='o', label=pass_k_col)
            
            plt.title(f'pass@k vs. Temperature for {model} on {dataset}')
            plt.xlabel('Temperature')
            plt.ylabel('pass@k Score')
            plt.grid(True)
            plt.legend()
            filename = output_dir / f'plot_{model}_{dataset}_pass_k_vs_temp.png'
            plt.savefig(filename)
            plt.close()
            print(f"Saved plot: {filename}")

def plot_model_comparison(df, output_dir):
    """Compares Base vs. Instruct models for each dataset and temperature."""
    for dataset in df['dataset'].unique():
        for temp in df['temperature'].unique():
            subset = df[(df['dataset'] == dataset) & (df['temperature'] == temp)]
            if subset.empty or len(subset['model'].unique()) < 2:
                continue
            
            pass_k_cols = sorted([col for col in subset.columns if col.startswith('pass@')], key=lambda x: int(x.split('@')[1]))
            metrics_to_plot = ['maj@1'] + pass_k_cols
            plot_data = subset.melt(id_vars='model', value_vars=metrics_to_plot, var_name='metric', value_name='score')

            plt.figure(figsize=(12, 7))
            sns.barplot(data=plot_data, x='metric', y='score', hue='model')
            plt.title(f'Model Comparison on {dataset} (Temp: {temp})')
            plt.xlabel('Metric')
            plt.ylabel('Score')
            plt.xticks(rotation=45)
            plt.tight_layout()
            filename = output_dir / f'plot_{dataset}_temp_{temp}_model_comparison.png'
            plt.savefig(filename)
            plt.close()
            print(f"Saved plot: {filename}")

def plot_across_datasets(df, output_dir):
    """Compares performance across datasets for each model and temperature."""
    for model in df['model'].unique():
        for temp in df['temperature'].unique():
            subset = df[(df['model'] == model) & (df['temperature'] == temp)]
            if subset.empty:
                continue

            pass_k_cols = sorted([col for col in subset.columns if col.startswith('pass@')], key=lambda x: int(x.split('@')[1]))
            metrics_to_plot = ['maj@1'] + pass_k_cols
            plot_data = subset.melt(id_vars='dataset', value_vars=metrics_to_plot, var_name='metric', value_name='score')

            # Using catplot which creates its own figure, so we don't create one with plt.figure
            g = sns.catplot(data=plot_data, x='metric', y='score', hue='dataset', kind='bar', aspect=2)
            g.fig.suptitle(f'Performance on {model} (Temp: {temp}) Across Datasets')
            g.set_xticklabels(rotation=45)
            # Adjust layout to prevent title overlap
            g.fig.tight_layout(rect=[0, 0, 1, 0.96])
            filename = output_dir / f'plot_{model}_temp_{temp}_dataset_comparison.png'
            plt.savefig(filename)
            plt.close()
            print(f"Saved plot: {filename}")


def plot_heatmaps(df, output_dir):
    """Generates heatmaps of pass@k and maj@1 scores."""
    pass_k_cols = sorted([col for col in df.columns if col.startswith('pass@')], key=lambda x: int(x.split('@')[1]))
    metrics_to_plot = ['maj@1'] + pass_k_cols

    for metric in metrics_to_plot:
        if metric not in df.columns:
            continue
        
        # Ensure there are values to pivot
        if df[metric].isnull().all():
            print(f"Skipping heatmap for {metric} as all values are null.")
            continue

        pivot_table = df.pivot_table(
            index=['model', 'dataset'],
            columns='temperature',
            values=metric
        )
        
        if pivot_table.empty:
            continue

        plt.figure(figsize=(10, 8))
        sns.heatmap(pivot_table, annot=True, fmt=".3f", cmap="viridis")
        plt.title(f'Heatmap of {metric} Scores')
        plt.tight_layout()
        filename = output_dir / f'heatmap_{metric.replace("@", "")}.png'
        plt.savefig(filename)
        plt.close()
        print(f"Saved plot: {filename}")


# --- Core Evaluation Logic (adapted from evaluate.py) ---

def process_and_evaluate_file(input_file, output_file, dataset_name):
    """
    Reads a JSONL file of generations, computes scores, saves the enhanced
    results, and returns the calculated pass@k and maj@1 metrics.
    """
    problem2scores = defaultdict(list)
    problem2maj_data = defaultdict(list)

    input_path = Path(input_file)
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    lines_processed = 0
    try:
        with open(input_path, "r", encoding="utf-8") as f:
            total_lines = sum(1 for _ in f)
    except FileNotFoundError:
        print(f"Error: Input file not found at: {input_file}")
        return {}

    with open(input_path, "r", encoding="utf-8") as f_in, \
         open(output_path, "w", encoding="utf-8") as f_out:

        for line_idx, line in enumerate(tqdm(f_in, total=total_lines, desc=f"Scoring {input_path.name}")):
            data = json.loads(line)
            lines_processed += 1

            model_answer = data.get("answer", "")
            gold_answer = data.get("gold", "")
            problem_id = data.get("id", line_idx)

            try:
                score_dict = compute_score(model_answer, gold_answer)
                data.update(score_dict)

                if score_dict:
                    score_val = float(score_dict.get("score", 0.0))
                    pred_val = score_dict.get("extracted_pred", None)
                    problem2scores[problem_id].append(score_val)
                    problem2maj_data[problem_id].append((pred_val, score_val))

            except Exception as e:
                tqdm.write(f"Error processing line {line_idx + 1}: {e}")
                data["error"] = str(e)

            f_out.write(json.dumps(data) + "\n")

    if lines_processed == 0:
        return {}

    # --- Pass@k Calculation ---
    k2problem_pass_vals = defaultdict(list)
    for problem_id, scores in problem2scores.items():
        n_resps = len(scores)
        if n_resps == 0: continue
        c_correct = sum(1 for score in scores if score > 0.0)
        
        # Determine k values based on the dataset
        if dataset_name.lower() in ["amc23", "aime25"]:
            possible_ks = [1, 2, 4, 8, 16, 32, 64]
        else: # Default for math500
            possible_ks = [1, 2, 4, 8, 16]
        
        ks = sorted(list(set([k for k in possible_ks if k <= n_resps])))

        for k in ks:
            try:
                pass_at_k = 1.0 - math.comb(n_resps - c_correct, k) / math.comb(n_resps, k)
            except (ValueError, ZeroDivisionError):
                pass_at_k = 0.0
            k2problem_pass_vals[k].append(pass_at_k)

    # --- Majority Vote Calculation ---
    maj_correct_count = 0
    maj_total = 0
    for problem_id, data_list in problem2maj_data.items():
        if not data_list: continue
        maj_total += 1
        
        preds = [item[0] for item in data_list]
        counts = Counter(preds)
        max_freq = max(counts.values())
        candidates = [p for p, c in counts.items() if c == max_freq]
        
        if len(candidates) > 1:
            # Sort to ensure deterministic tie-breaking
            candidates.sort(key=lambda x: str(x))
        winner = random.choice(candidates)
        
        is_winner_correct = any(score > 0.0 for pred, score in data_list if pred == winner)
        if is_winner_correct:
            maj_correct_count += 1

    # --- Package and Return Results ---
    metrics = {}
    print(f"\n--- Metrics for {input_path.name} ---")
    if k2problem_pass_vals:
        print("Pass@k Metrics:")
        for k in sorted(k2problem_pass_vals.keys()):
            vals = k2problem_pass_vals[k]
            if not vals: continue
            avg_pass = sum(vals) / len(vals)
            metrics[f'pass@{k}'] = avg_pass
            print(f"  pass@{k:<4}: {avg_pass*100:.2f}%")

    if maj_total > 0:
        maj_acc = maj_correct_count / maj_total
        metrics['maj@1'] = maj_acc
        print("\nMajority Vote Metric:")
        print(f"  maj@1    : {maj_acc*100:.2f}%")
    
    print(f"Scored results saved to {output_file}\n")
    return metrics


# --- Main Orchestration Function ---

def main():
    parser = argparse.ArgumentParser(description="Run evaluation, aggregate results, and generate plots.")
    parser.add_argument("--input_dir", type=str, default="outputs", help="Directory with generation files.")
    parser.add_argument("--output_dir", type=str, default="outputs", help="Directory to save summaries and plots.")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(exist_ok=True)

    # --- Stage 1: Find and Evaluate Raw Generation Files ---
    print("--- Stage 1: Evaluating generation files ---")
    all_files = list(input_dir.glob("*.jsonl"))
    generation_files = sorted([
        f for f in all_files
        if not f.name.endswith(("_eval.jsonl", ".rank0.jsonl", ".rank1.jsonl"))
    ])

    if not generation_files:
        print("Warning: No raw generation files found to evaluate. Halting.")
        return

    print(f"Found {len(generation_files)} raw generation files to process.")
    
    all_results_agg = []

    for gen_file in generation_files:
        match = re.match(r"(math500|amc23|aime25)_(base|instruct)_t(\d\.\d+)", gen_file.stem)
        if not match:
            print(f"Warning: Skipping file with unexpected name format: {gen_file.name}")
            continue
        
        dataset, model_type, temp_str = match.groups()
        
        eval_file = gen_file.with_name(gen_file.stem + "_eval.jsonl")
        
        # Run evaluation and get metrics
        metrics = process_and_evaluate_file(str(gen_file), str(eval_file), dataset)

        if metrics:
            result_row = {
                "model": "Instruct" if model_type == "instruct" else "Base",
                "dataset": dataset.upper() if dataset != 'math500' else 'Math500',
                "temperature": float(temp_str),
            }
            result_row.update(metrics)
            all_results_agg.append(result_row)

    if not all_results_agg:
        print("Error: No valid results could be aggregated. Halting analysis.")
        return

    # --- Stage 2: Save Summaries and Generate Plots ---
    print("\n--- Stage 2: Aggregating results and generating reports ---")
    df = pd.DataFrame(all_results_agg)
    df = df.sort_values(by=["model", "dataset", "temperature"]).reset_index(drop=True)

    # Save detailed JSON
    json_summary_path = output_dir / "all_results.json"
    df.to_json(json_summary_path, orient='records', indent=4)
    print(f"Saved detailed summary to: {json_summary_path}")

    # Save CSV
    csv_summary_path = output_dir / "summary.csv"
    df.to_csv(csv_summary_path, index=False)
    print(f"Saved CSV summary to: {csv_summary_path}")

    # Generate Plots
    print("\n--- Generating Plots ---")
    plot_pass_at_k_by_temperature(df, plots_dir)
    plot_model_comparison(df, plots_dir)
    plot_across_datasets(df, plots_dir)
    plot_heatmaps(df, plots_dir)
    print("\n--- Analysis Complete ---")


if __name__ == "__main__":
    main()
