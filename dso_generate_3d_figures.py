#!/usr/bin/env python3
"""
Generate 3D figures showing R² vs sample size for each policy and dataset.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import Patch
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
        parts = dir_name.replace('dso_experiments_', '').split('_')

        # Determine dataset, samples, and policy
        if parts[0] in ['abg', 'ci', 'indoor', 'outdoor']:
            dataset = parts[0]
            samples_str = parts[1]
            policy = '_'.join(parts[2:])
        elif parts[0].endswith('k'):
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
            }
        except Exception as e:
            print(f"Error loading {summary_path}: {e}")
            continue

    return results


def create_3d_figure(results, dataset, output_path):
    """Create a 3D scatter plot showing R² vs sample size for each policy."""

    if dataset not in results:
        print(f"No results found for dataset: {dataset}")
        return

    dataset_results = results[dataset]

    # Get all sample sizes and sort them in descending order (reversed)
    samples_list = sorted(dataset_results.keys(), reverse=True)
    # Policy order: RSPG, PQT, VPG
    policies = ['risk_seeking', 'priority_queue', 'vanilla']
    policy_labels = {'risk_seeking': 'RSPG', 'priority_queue': 'PQT', 'vanilla': 'VPG'}
    policy_colors = {'risk_seeking': '#e74c3c', 'priority_queue': '#2ecc71', 'vanilla': '#3498db'}
    policy_markers = {'risk_seeking': 'o', 'priority_queue': '^', 'vanilla': 's'}

    # Create figure
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Create scatter plots for each policy
    for policy_idx, policy in enumerate(policies):
        xs = []
        ys = []
        zs = []

        for sample_idx, samples in enumerate(samples_list):
            if policy in dataset_results[samples]:
                r2 = dataset_results[samples][policy].get('r2_mean', np.nan)
                if not np.isnan(r2):
                    xs.append(sample_idx)
                    ys.append(policy_idx)
                    zs.append(r2)

        if xs:
            ax.scatter(xs, ys, zs, c=policy_colors[policy], marker=policy_markers[policy],
                      s=100, label=policy_labels[policy], alpha=0.8)

    # Set labels with doubled font sizes
    ax.set_xlabel('Sample Size', fontsize=20, labelpad=15)
    ax.set_ylabel('Policy', fontsize=20, labelpad=15)
    ax.set_zlabel('R²', fontsize=20, labelpad=15)
    ax.set_title(f'{dataset.upper()} Dataset: R² vs Sample Size by Policy', fontsize=20)

    # Set ticks - show only every second label to avoid overlap
    ax.set_xticks(range(len(samples_list)))
    x_labels = [f'{s//1000}k' if i % 2 == 0 else '' for i, s in enumerate(samples_list)]
    ax.set_xticklabels(x_labels, fontsize=16)
    ax.set_yticks(range(len(policies)))
    ax.set_yticklabels([policy_labels[p] for p in policies], fontsize=16)
    ax.set_zlim(0, 1.0)
    ax.tick_params(axis='z', labelsize=16)

    # Add legend with doubled font size
    ax.legend(loc='upper left', fontsize=18)

    # Adjust view angle
    ax.view_init(elev=20, azim=45)

    plt.tight_layout()

    # Save figure
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Figure saved to {output_path}")

    # Also save as PNG if PDF
    if output_path.endswith('.pdf'):
        png_path = output_path.replace('.pdf', '.png')
        plt.savefig(png_path, dpi=150, bbox_inches='tight')
        print(f"Figure saved to {png_path}")

    plt.close()


def create_line_3d_figure(results, dataset, output_path):
    """Create a 3D line plot showing R² trends across sample sizes."""

    if dataset not in results:
        print(f"No results found for dataset: {dataset}")
        return

    dataset_results = results[dataset]

    # Get all sample sizes and sort them in descending order (reversed)
    samples_list = sorted(dataset_results.keys(), reverse=True)
    # Policy order: RSPG, PQT, VPG
    policies = ['risk_seeking', 'priority_queue', 'vanilla']
    policy_labels = {'risk_seeking': 'RSPG', 'priority_queue': 'PQT', 'vanilla': 'VPG'}
    policy_colors = {'risk_seeking': '#e74c3c', 'priority_queue': '#2ecc71', 'vanilla': '#3498db'}
    policy_markers = {'risk_seeking': 'o', 'priority_queue': '^', 'vanilla': 's'}

    # Create figure
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Plot lines for each policy
    for policy_idx, policy in enumerate(policies):
        xs = []
        ys = []
        zs = []

        for sample_idx, samples in enumerate(samples_list):
            if policy in dataset_results[samples]:
                r2 = dataset_results[samples][policy].get('r2_mean', np.nan)
                if not np.isnan(r2):
                    xs.append(sample_idx)
                    ys.append(policy_idx)
                    zs.append(r2)

        if xs:
            ax.plot(xs, ys, zs, color=policy_colors[policy], linewidth=2,
                   marker=policy_markers[policy], markersize=8, label=policy_labels[policy])

            # Add error bars (vertical lines for std)
            for i, (x, y, z) in enumerate(zip(xs, ys, zs)):
                samples = samples_list[x]
                if policy in dataset_results[samples]:
                    std = dataset_results[samples][policy].get('r2_std', 0)
                    ax.plot([x, x], [y, y], [z-std, z+std], color=policy_colors[policy], alpha=0.5, linewidth=1)

    # Set labels with doubled font sizes
    ax.set_xlabel('Sample Size', fontsize=20, labelpad=15)
    ax.set_ylabel('Policy', fontsize=20, labelpad=15)
    ax.set_zlabel('R²', fontsize=20, labelpad=15)
    ax.set_title(f'{dataset.upper()} Dataset: R² vs Sample Size by Policy', fontsize=20)

    # Set ticks - show only every second label to avoid overlap
    ax.set_xticks(range(len(samples_list)))
    x_labels = [f'{s//1000}k' if i % 2 == 0 else '' for i, s in enumerate(samples_list)]
    ax.set_xticklabels(x_labels, fontsize=16)
    ax.set_yticks(range(len(policies)))
    ax.set_yticklabels([policy_labels[p] for p in policies], fontsize=16)
    ax.set_zlim(0, 1.0)
    ax.tick_params(axis='z', labelsize=16)

    # Add legend with doubled font size
    ax.legend(loc='upper left', fontsize=18)

    # Adjust view angle
    ax.view_init(elev=25, azim=45)

    plt.tight_layout()

    # Save figure
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Figure saved to {output_path}")

    # Also save as PNG if PDF
    if output_path.endswith('.pdf'):
        png_path = output_path.replace('.pdf', '.png')
        plt.savefig(png_path, dpi=150, bbox_inches='tight')
        print(f"Figure saved to {png_path}")

    plt.close()


def main():
    """Main function to generate 3D figures."""
    print("Loading experiment results...")
    results = load_experiment_results('.')

    if not results:
        print("No results found!")
        return

    # Print summary of loaded data
    for dataset in results:
        samples_list = sorted(results[dataset].keys())
        print(f"\n{dataset.upper()}: {len(samples_list)} sample sizes")
        print(f"  Samples: {[f'{s//1000}k' for s in samples_list]}")

    # Create output directory
    os.makedirs('figures', exist_ok=True)

    # Generate 3D figures for each dataset
    datasets = ['abg', 'ci', 'indoor', 'outdoor']

    for dataset in datasets:
        if dataset in results:
            print(f"\nGenerating 3D figure for {dataset.upper()}...")
            create_3d_figure(results, dataset, f'figures/r2_3d_{dataset}.pdf')
            create_line_3d_figure(results, dataset, f'figures/r2_3d_line_{dataset}.pdf')

    print("\nDone!")


if __name__ == '__main__':
    main()
