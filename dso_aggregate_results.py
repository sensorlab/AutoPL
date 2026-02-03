#!/usr/bin/env python3
"""
Aggregate DSO experiment results and generate LaTeX tables and figures.

This script:
1. Collects results from all experiment directories
2. Generates LaTeX table rows for each dataset
3. Creates a figure showing R² vs sample size for each policy

Usage:
    python aggregate_results.py
"""

import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict


def load_experiment_results(base_dir='.'):
    """Load all experiment results from summary.csv files."""
    results = defaultdict(lambda: defaultdict(dict))

    for dir_name in os.listdir(base_dir):
        if not dir_name.startswith('dso_experiments_'):
            continue

        summary_path = os.path.join(base_dir, dir_name, 'results', 'summary.csv')
        if not os.path.exists(summary_path):
            continue

        # Parse directory name to extract dataset, samples, and policy
        # Format: dso_experiments_<dataset>_<samples>k_<policy>
        # or: dso_experiments_<samples>k_<policy> (legacy ABG format)
        parts = dir_name.replace('dso_experiments_', '').split('_')

        # Determine dataset, samples, and policy
        if parts[0] in ['abg', 'ci', 'indoor', 'outdoor']:
            dataset = parts[0]
            samples_str = parts[1]  # e.g., "10k"
            policy = '_'.join(parts[2:])  # e.g., "risk_seeking" or "priority_queue"
        elif parts[0].endswith('k'):
            # Legacy format: dso_experiments_10k_risk_seeking (ABG)
            dataset = 'abg'
            samples_str = parts[0]
            policy = '_'.join(parts[1:])
        else:
            continue

        # Parse samples (e.g., "10k" -> 10000)
        try:
            samples = int(samples_str.replace('k', '')) * 1000
        except ValueError:
            continue

        # Load summary CSV
        try:
            df = pd.read_csv(summary_path)
            if len(df) == 0:
                continue
            row = df.iloc[0]

            results[dataset][samples][policy] = {
                'mae_mean': row.get('mae_mean', np.nan),
                'mae_std': row.get('mae_std', np.nan),
                'mse_mean': row.get('mse_mean', np.nan),
                'mse_std': row.get('mse_std', np.nan),
                'mape_mean': row.get('mape_mean', np.nan),
                'mape_std': row.get('mape_std', np.nan),
                'r2_mean': row.get('r2_mean', np.nan),
                'r2_std': row.get('r2_std', np.nan),
                'best_expression': row.get('best_expression', ''),
                'n_runs': row.get('n_runs', 10),
                'dir_name': dir_name
            }
        except Exception as e:
            print(f"Error loading {summary_path}: {e}")
            continue

    return results


def format_latex_value(mean, std, decimals=2):
    """Format a value with std for LaTeX."""
    if pd.isna(mean) or pd.isna(std):
        return "N/A"
    return f"{mean:.{decimals}f} \\par $\\pm${std:.{decimals}f}"


def format_expression_for_latex(expr):
    """Format a symbolic expression for LaTeX (basic formatting)."""
    if not expr or pd.isna(expr):
        return "N/A"
    # Truncate if too long
    if len(str(expr)) > 80:
        return str(expr)[:77] + "..."
    return str(expr)


def generate_latex_table_rows(results, dataset, samples_list=None):
    """Generate LaTeX table rows for a specific dataset."""
    if samples_list is None:
        samples_list = [10000, 20000, 30000, 50000]

    policy_names = {
        'risk_seeking': 'DSR-RSPG',
        'vanilla': 'DSR-VPG',
        'priority_queue': 'DSR-PQT'
    }

    latex_rows = []

    if dataset not in results:
        print(f"No results found for dataset: {dataset}")
        return ""

    dataset_results = results[dataset]

    # Find the best sample size for each policy (highest R²)
    best_results = {}
    for policy in ['risk_seeking', 'vanilla', 'priority_queue']:
        best_r2 = -1
        best_sample = None
        for samples in samples_list:
            if samples in dataset_results and policy in dataset_results[samples]:
                r2 = dataset_results[samples][policy].get('r2_mean', -1)
                if r2 > best_r2:
                    best_r2 = r2
                    best_sample = samples
        if best_sample:
            best_results[policy] = dataset_results[best_sample][policy]
            best_results[policy]['samples'] = best_sample

    # Generate rows for each policy
    for policy in ['risk_seeking', 'priority_queue', 'vanilla']:
        if policy not in best_results:
            continue

        r = best_results[policy]
        samples_k = r['samples'] // 1000

        row = f"""% {policy_names[policy]} ({samples_k}k samples)
\\textbf{{{policy_names[policy]}}} &
{format_latex_value(r['mae_mean'], r['mae_std'])} &
{format_latex_value(r['mse_mean'], r['mse_std'])} &
{format_latex_value(r['mape_mean'], r['mape_std'])} &
{format_latex_value(r['r2_mean'], r['r2_std'])} &
\\centering {format_expression_for_latex(r['best_expression'])} &
High & ? \\\\ \\hline
"""
        latex_rows.append(row)

    return '\n'.join(latex_rows)


def generate_full_latex_table(results, dataset, samples_list=None):
    """Generate a complete LaTeX table for a dataset."""
    if samples_list is None:
        samples_list = [10000, 20000, 30000, 50000]

    dataset_upper = dataset.upper()

    header = f"""% LaTeX table for {dataset_upper} dataset
% Generated by aggregate_results.py

\\begin{{tabular}}{{|m{{2.5cm}}|
                >{{\\centering\\arraybackslash}}m{{1.3cm}}|
                >{{\\centering\\arraybackslash}}m{{1.3cm}}|
                >{{\\centering\\arraybackslash}}m{{1.3cm}}|
                >{{\\centering\\arraybackslash}}m{{1.3cm}}|
                m{{5.5cm}}|
                >{{\\centering\\arraybackslash}}m{{1.8cm}}|
                >{{\\centering\\arraybackslash}}m{{1.2cm}}|}}
\\hline

% Header Row
\\diagbox{{\\textbf{{Methods}}}}{{\\textbf{{Metrics}}}} &
\\textbf{{MAE}} \\par $\\pm$\\textbf{{STD}} &
\\textbf{{MSE}} \\par $\\pm$\\textbf{{STD}} &
\\textbf{{MAPE}} \\par $\\pm$\\textbf{{STD}} &
\\textbf{{R\\textsuperscript{{2}}}} \\par $\\pm$\\textbf{{STD}} &
\\centering \\textbf{{Expression}} &
\\textbf{{Interpret.}} &
\\textbf{{Validity}} \\\\ \\hline \\hline

"""

    rows = generate_latex_table_rows(results, dataset, samples_list)

    footer = """
\\end{tabular}
"""

    return header + rows + footer


def create_r2_vs_samples_figure(results, output_path='figures/r2_vs_samples.pdf'):
    """Create a figure showing R² vs sample size for each policy and dataset."""

    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)

    datasets = ['abg', 'ci', 'indoor', 'outdoor']
    policies = ['risk_seeking', 'vanilla', 'priority_queue']
    policy_labels = {'risk_seeking': 'Risk-Seeking PG', 'vanilla': 'Vanilla PG', 'priority_queue': 'Priority Queue'}
    policy_colors = {'risk_seeking': '#e74c3c', 'vanilla': '#3498db', 'priority_queue': '#2ecc71'}
    policy_markers = {'risk_seeking': 'o', 'vanilla': 's', 'priority_queue': '^'}

    samples_list = [10000, 20000, 30000, 50000]
    samples_labels = ['10k', '20k', '30k', '50k']

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('R² vs Sample Size by Policy and Dataset', fontsize=14, fontweight='bold')

    for idx, dataset in enumerate(datasets):
        ax = axes[idx // 2, idx % 2]

        if dataset not in results:
            ax.set_title(f'{dataset.upper()} (No data)')
            continue

        dataset_results = results[dataset]

        for policy in policies:
            x_vals = []
            y_vals = []
            y_errs = []

            for i, samples in enumerate(samples_list):
                if samples in dataset_results and policy in dataset_results[samples]:
                    r = dataset_results[samples][policy]
                    x_vals.append(i)
                    y_vals.append(r['r2_mean'])
                    y_errs.append(r['r2_std'])

            if x_vals:
                ax.errorbar(x_vals, y_vals, yerr=y_errs,
                           label=policy_labels[policy],
                           color=policy_colors[policy],
                           marker=policy_markers[policy],
                           markersize=8,
                           capsize=4,
                           linewidth=2)

        ax.set_xticks(range(len(samples_labels)))
        ax.set_xticklabels(samples_labels)
        ax.set_xlabel('Number of Samples')
        ax.set_ylabel('R² Score')
        ax.set_title(f'{dataset.upper()} Dataset')
        ax.legend(loc='lower right')
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1.05)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Figure saved to {output_path}")

    # Also save as PNG
    png_path = output_path.replace('.pdf', '.png')
    plt.savefig(png_path, dpi=150, bbox_inches='tight')
    print(f"Figure saved to {png_path}")

    plt.close()


def create_combined_figure(results, output_path='figures/r2_combined.pdf'):
    """Create a combined figure with all datasets on one plot."""

    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)

    datasets = ['abg', 'ci', 'indoor', 'outdoor']
    policies = ['risk_seeking', 'vanilla', 'priority_queue']

    samples_list = [10000, 20000, 30000, 50000]
    samples_labels = ['10k', '20k', '30k', '50k']

    # Create figure with subplots for each policy
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle('R² vs Sample Size: Comparison Across Datasets', fontsize=14, fontweight='bold')

    policy_titles = {'risk_seeking': 'Risk-Seeking PG', 'vanilla': 'Vanilla PG', 'priority_queue': 'Priority Queue'}
    dataset_colors = {'abg': '#e74c3c', 'ci': '#3498db', 'indoor': '#2ecc71', 'outdoor': '#9b59b6'}
    dataset_markers = {'abg': 'o', 'ci': 's', 'indoor': '^', 'outdoor': 'D'}

    for policy_idx, policy in enumerate(policies):
        ax = axes[policy_idx]

        for dataset in datasets:
            if dataset not in results:
                continue

            dataset_results = results[dataset]
            x_vals = []
            y_vals = []
            y_errs = []

            for i, samples in enumerate(samples_list):
                if samples in dataset_results and policy in dataset_results[samples]:
                    r = dataset_results[samples][policy]
                    x_vals.append(i)
                    y_vals.append(r['r2_mean'])
                    y_errs.append(r['r2_std'])

            if x_vals:
                ax.errorbar(x_vals, y_vals, yerr=y_errs,
                           label=dataset.upper(),
                           color=dataset_colors[dataset],
                           marker=dataset_markers[dataset],
                           markersize=8,
                           capsize=4,
                           linewidth=2)

        ax.set_xticks(range(len(samples_labels)))
        ax.set_xticklabels(samples_labels)
        ax.set_xlabel('Number of Samples')
        ax.set_ylabel('R² Score')
        ax.set_title(policy_titles[policy])
        ax.legend(loc='lower right')
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1.05)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Figure saved to {output_path}")

    png_path = output_path.replace('.pdf', '.png')
    plt.savefig(png_path, dpi=150, bbox_inches='tight')
    print(f"Figure saved to {png_path}")

    plt.close()


def print_summary_table(results):
    """Print a summary table of all results."""
    print("\n" + "=" * 100)
    print("SUMMARY OF ALL EXPERIMENTS")
    print("=" * 100)

    datasets = ['abg', 'ci', 'indoor', 'outdoor']
    samples_list = [10000, 20000, 30000, 50000]
    policies = ['risk_seeking', 'vanilla', 'priority_queue']

    for dataset in datasets:
        if dataset not in results:
            continue

        print(f"\n{dataset.upper()} Dataset:")
        print("-" * 80)
        print(f"{'Samples':<10} {'Policy':<15} {'R² Mean':<12} {'R² Std':<10} {'MAE':<10} {'MAPE':<10}")
        print("-" * 80)

        for samples in samples_list:
            if samples not in results[dataset]:
                continue

            for policy in policies:
                if policy not in results[dataset][samples]:
                    continue

                r = results[dataset][samples][policy]
                samples_str = f"{samples//1000}k"
                print(f"{samples_str:<10} {policy:<15} {r['r2_mean']:<12.4f} {r['r2_std']:<10.4f} {r['mae_mean']:<10.2f} {r['mape_mean']:<10.2f}%")


def save_latex_tables(results, output_dir='latex_tables'):
    """Save LaTeX tables for all datasets."""
    os.makedirs(output_dir, exist_ok=True)

    datasets = ['abg', 'ci', 'indoor', 'outdoor']

    for dataset in datasets:
        if dataset not in results:
            continue

        # Generate full table
        latex_table = generate_full_latex_table(results, dataset)

        output_path = os.path.join(output_dir, f'{dataset}_table.tex')
        with open(output_path, 'w') as f:
            f.write(latex_table)
        print(f"LaTeX table saved to {output_path}")

        # Also generate just the DSR rows for easy insertion
        rows_only = generate_latex_table_rows(results, dataset)
        rows_path = os.path.join(output_dir, f'{dataset}_dsr_rows.tex')
        with open(rows_path, 'w') as f:
            f.write(f"% DSR rows for {dataset.upper()} dataset\n")
            f.write(f"% Insert these rows into your existing table\n\n")
            f.write(rows_only)
        print(f"LaTeX rows saved to {rows_path}")


def main():
    """Main function to aggregate results and generate outputs."""
    print("Loading experiment results...")
    results = load_experiment_results('.')

    if not results:
        print("No results found!")
        return

    # Print summary
    print_summary_table(results)

    # Save LaTeX tables
    print("\nGenerating LaTeX tables...")
    save_latex_tables(results)

    # Create figures
    print("\nGenerating figures...")
    create_r2_vs_samples_figure(results)
    create_combined_figure(results)

    print("\nDone!")


if __name__ == '__main__':
    main()
