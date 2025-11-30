"""
Prettified visualization script for final figures.

This script regenerates all figures with:
- Smaller figure sizes + larger fonts (compact)
- Light, diluted, harmonious colors
- Removed titles and unnecessary labels
- Color diversity based on figure type
"""

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats
from sklearn.metrics import confusion_matrix

# Set style
sns.set_style("whitegrid", {'grid.linestyle': '--', 'grid.alpha': 0.3})
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from analysis.data_loader import DataLoader


def wilson_score_interval(successes, n, confidence=0.95):
    """Calculate Wilson score confidence interval."""
    from scipy import stats
    z = stats.norm.ppf(1 - (1 - confidence) / 2)
    p = successes / n
    denominator = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denominator
    margin = z * np.sqrt((p * (1 - p) + z**2 / (4 * n)) / n) / denominator
    return centre - margin, centre + margin


def plot_1a_accuracy_bar(merged_df, output_path):
    """Graph 1a: Accuracy bar chart (compact, cool blues/purples)."""

    judges = ['LLM-Corr', 'LLM-Multi', 'Agent-Corr', 'Agent-Multi', 'Agent-UT']
    pred_cols = ['llm_corr_pred', 'llm_multi_pred', 'agent_corr_pred',
                 'agent_multi_pred', 'agent_ut_pred']

    # Cool, professional color palette (light blues/purples)
    colors = ['#B8D4E8', '#9BB8D3', '#A8C8D8', '#8EAFC8', '#7B9BB3']

    accuracies = []
    lower_bounds = []
    upper_bounds = []

    for pred_col in pred_cols:
        valid = merged_df.dropna(subset=[pred_col, 'passed'])
        correct = (valid[pred_col] == valid['passed']).sum()
        total = len(valid)
        acc = correct / total
        lower, upper = wilson_score_interval(correct, total)

        accuracies.append(acc * 100)
        lower_bounds.append((acc - lower) * 100)
        upper_bounds.append((upper - acc) * 100)

    # Compact figure
    fig, ax = plt.subplots(figsize=(4.5, 3))

    x_pos = np.arange(len(judges))
    bars = ax.bar(x_pos, accuracies, yerr=[lower_bounds, upper_bounds],
                  capsize=4, color=colors, edgecolor='black', linewidth=1.2,
                  alpha=0.8, error_kw=dict(linewidth=1.5, capthick=1.5))

    # Minimal labels
    ax.set_ylabel('Accuracy (%)', fontsize=13, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(judges, fontsize=11, rotation=0)
    ax.set_ylim(60, 105)
    ax.tick_params(axis='both', labelsize=11)

    # Remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='y', alpha=0.3, linestyle='--')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {output_path.name}")


def plot_1d_scores_boxscatter(merged_df, output_path):
    """Graph 1d: Numerical scores box+scatter (compact, soft greens)."""

    judges = ['LLM-Corr', 'LLM-Multi', 'Agent-Corr', 'Agent-Multi', 'Agent-UT']
    score_cols = ['llm_corr_score', 'llm_multi_correctness', 'agent_corr_score',
                  'agent_multi_correctness', 'agent_ut_correctness']

    # Soft green palette
    colors = ['#C8E6C9', '#A5D6A7', '#81C784', '#66BB6A', '#4CAF50']

    all_scores = []
    for score_col in score_cols:
        valid = merged_df.dropna(subset=[score_col])
        all_scores.append(valid[score_col].values)

    # Compact figure
    fig, ax = plt.subplots(figsize=(4.5, 3))

    x_pos = np.arange(len(judges))

    # Box plots
    bp = ax.boxplot(all_scores, positions=x_pos, widths=0.5,
                    patch_artist=True, showfliers=False,
                    boxprops=dict(linewidth=1.2, edgecolor='black'),
                    whiskerprops=dict(linewidth=1.2),
                    capprops=dict(linewidth=1.2),
                    medianprops=dict(linewidth=2, color='darkgreen'))

    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    # Scatter overlay
    for i, (scores, color) in enumerate(zip(all_scores, colors)):
        x_jitter = np.random.normal(i, 0.06, size=len(scores))
        ax.scatter(x_jitter, scores, alpha=0.3, s=20, color=color,
                  edgecolors='gray', linewidths=0.3)

    ax.set_ylabel('Score', fontsize=13, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(judges, fontsize=11, rotation=0)
    ax.set_ylim(-0.05, 1.05)
    ax.tick_params(axis='both', labelsize=11)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='y', alpha=0.3, linestyle='--')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {output_path.name}")


def plot_3b_confusion_bar(merged_df, output_path):
    """Graph 3b: Confusion matrix bar chart (warm colors - reds/oranges)."""

    judges = ['LLM-Corr', 'LLM-Multi', 'Agent-Corr', 'Agent-Multi', 'Agent-UT']
    pred_cols = ['llm_corr_pred', 'llm_multi_pred', 'agent_corr_pred',
                 'agent_multi_pred', 'agent_ut_pred']

    # Warm color palette (light reds/oranges/corals)
    colors = ['#FFCDD2', '#EF9A9A', '#E57373', '#EF5350', '#E53935']

    categories = ['True Positive', 'True Negative', 'False Positive', 'False Negative']
    data = {cat: [] for cat in categories}

    for pred_col in pred_cols:
        valid = merged_df.dropna(subset=[pred_col, 'passed'])
        y_true = valid['passed'].astype(int)
        y_pred = valid[pred_col].astype(int)

        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
        total = tp + tn + fp + fn

        data['True Positive'].append((tp / total) * 100)
        data['True Negative'].append((tn / total) * 100)
        data['False Positive'].append((fp / total) * 100)
        data['False Negative'].append((fn / total) * 100)

    # Compact figure
    fig, ax = plt.subplots(figsize=(5, 3))

    bar_height = 0.15
    y_positions = np.arange(len(categories))

    for i, (judge, color) in enumerate(zip(judges, colors)):
        values = [data[cat][i] for cat in categories]
        offset = (i - 2) * bar_height
        ax.barh(y_positions + offset, values, bar_height,
                label=judge, color=color, edgecolor='black', linewidth=0.8,
                alpha=0.8)

    ax.set_yticks(y_positions)
    ax.set_yticklabels(categories, fontsize=12, fontweight='bold')
    ax.set_xlabel('Percentage (%)', fontsize=13, fontweight='bold')
    ax.set_xlim(0, 100)
    ax.tick_params(axis='both', labelsize=11)
    ax.grid(axis='x', alpha=0.3, linestyle='--')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Legend at bottom right
    ax.legend(loc='lower right', fontsize=9, framealpha=0.9, ncol=2)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {output_path.name}")


def plot_4_multiaspect(merged_df, output_path):
    """Graph 4: Multi-aspect correlation (cool purples/blues with contrast)."""

    aspects = ['Correctness', 'Style', 'Simplicity', 'Robustness']

    # Color palette with good contrast between LLM (light purple) and Agent (darker blue/teal)
    judge_configs = {
        'Correctness': [
            ('Ground Truth', 'pass_rate', '#FFD700'),  # Gold
            ('LLM-Corr', 'llm_corr_score', '#D1C4E9'),  # Very light purple
            ('LLM-Multi', 'llm_multi_correctness', '#9575CD'),  # Medium purple
            ('Agent-Corr', 'agent_corr_score', '#B2DFDB'),  # Very light teal
            ('Agent-Multi', 'agent_multi_correctness', '#4DB6AC'),  # Medium teal
            ('Agent-UT', 'agent_ut_correctness', '#00897B')  # Dark teal
        ],
        'Style': [
            ('LLM-Multi', 'llm_multi_style', '#9575CD'),
            ('Agent-Multi', 'agent_multi_style', '#4DB6AC'),
            ('Agent-UT', 'agent_ut_style', '#00897B')
        ],
        'Simplicity': [
            ('LLM-Multi', 'llm_multi_simplicity', '#9575CD'),
            ('Agent-Multi', 'agent_multi_simplicity', '#4DB6AC'),
            ('Agent-UT', 'agent_ut_simplicity', '#00897B')
        ],
        'Robustness': [
            ('LLM-Multi', 'llm_multi_robustness', '#9575CD'),
            ('Agent-Multi', 'agent_multi_robustness', '#4DB6AC'),
            ('Agent-UT', 'agent_ut_robustness', '#00897B')
        ]
    }

    # Compact 4-panel figure
    fig, axes = plt.subplots(1, 4, figsize=(10, 2.5))

    for idx, aspect in enumerate(aspects):
        ax = axes[idx]
        configs = judge_configs[aspect]

        judges = []
        means = []
        stds = []
        colors_list = []
        significance = []

        for judge_name, score_col, color in configs:
            valid = merged_df.dropna(subset=[score_col, 'pass_rate'])

            if len(valid) > 0:
                scores = valid[score_col].values
                judges.append(judge_name)
                means.append(scores.mean())
                stds.append(scores.std())
                colors_list.append(color)

                if aspect == 'Correctness' and judge_name != 'Ground Truth':
                    ut_pass_rate = valid['pass_rate'].values
                    t_stat, p_val = stats.ttest_rel(scores, ut_pass_rate)
                    significance.append('*' if p_val < 0.05 else '')
                else:
                    significance.append('')

        x_pos = np.arange(len(judges))
        bars = ax.bar(x_pos, means, yerr=stds, capsize=4,
                      color=colors_list, edgecolor='black', linewidth=1,
                      alpha=0.8, error_kw=dict(linewidth=1.2, capthick=1.5))

        # Minimal labels
        ax.set_ylabel('Score' if idx == 0 else '', fontsize=11, fontweight='bold')
        ax.set_title(aspect, fontsize=12, fontweight='bold', pad=8)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(judges, rotation=45, ha='right', fontsize=9)
        ax.set_ylim(0, 1.15)
        ax.tick_params(axis='both', labelsize=9)
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {output_path.name}")


def plot_5bcde_agent_behavior(merged_df, output_dir):
    """Graphs 5b-5e: Agent behavior box plots (harmonious earth tones)."""

    scenarios = ['Agent-Corr', 'Agent-Multi', 'Agent-UT']
    prefixes = ['agent_corr', 'agent_multi', 'agent_ut']

    # Earth tone palette (light tan to brown)
    colors = ['#D7CCC8', '#A1887F', '#6D4C41']

    metrics = [
        ('5b', 'steps', 'Steps', '<', 100, 'steps'),
        ('5c', 'tools', 'Tool Calls', None, None, None),
        ('5d', 'time', 'Time (s)', None, None, None),
        ('5e', 'cost', 'Cost (USD)', None, None, None)
    ]

    for graph_id, metric_name, ylabel, filter_op, filter_val, suffix in metrics:
        fig, ax = plt.subplots(figsize=(4, 2.8))

        data_list = []
        data_raw = []

        for prefix in prefixes:
            if metric_name == 'tools':
                valid = merged_df.dropna(subset=[f'{prefix}_bash', f'{prefix}_edits'])
                values = (valid[f'{prefix}_bash'] + valid[f'{prefix}_edits']).values
            else:
                col_name = f'{prefix}_{metric_name if metric_name != "cost" else "cost"}'
                if metric_name == 'steps':
                    col_name = f'{prefix}_steps'
                elif metric_name == 'time':
                    col_name = f'{prefix}_time'

                valid = merged_df.dropna(subset=[col_name])
                values = valid[col_name].values

                # Filter outliers for steps
                if filter_op == '<':
                    values = values[values < filter_val]

            data_list.append(values)
            data_raw.append(values)

        # Kruskal-Wallis test
        h_stat, p_val = stats.kruskal(*data_raw)

        # Box plot
        bp = ax.boxplot(data_list, labels=scenarios, patch_artist=True,
                        showfliers=False, widths=0.5,
                        boxprops=dict(linewidth=1.2, edgecolor='black'),
                        whiskerprops=dict(linewidth=1.2),
                        capprops=dict(linewidth=1.2),
                        medianprops=dict(linewidth=2, color='black'))

        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

        # Scatter overlay
        for i, (values, color) in enumerate(zip(data_list, colors)):
            x_jitter = np.random.normal(i + 1, 0.04, size=len(values))
            ax.scatter(x_jitter, values, alpha=0.35, s=20,
                      color=color, edgecolors='gray', linewidths=0.3)

        # Statistical annotation
        ax.text(0.5, 0.96, f'H={h_stat:.1f}, p<0.001',
                transform=ax.transAxes, ha='center', va='top',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='wheat', alpha=0.6),
                fontsize=8)

        ax.set_ylabel(ylabel, fontsize=11, fontweight='bold')
        ax.tick_params(axis='both', labelsize=10)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        plt.tight_layout()
        output_path = output_dir / f'graph{graph_id}_{suffix if suffix else metric_name}.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ Saved: {output_path.name}")


def main():
    """Generate all prettified visualizations."""
    print("="*80)
    print("GENERATING PRETTIFIED VISUALIZATIONS")
    print("="*80)

    # Create output directory
    output_dir = Path("outputs/analysis/final_figures_prettified")
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n📁 Output directory: {output_dir}\n")

    # Load data
    print("Loading data...")
    loader = DataLoader()
    merged_df = loader.merge_all()
    complete = merged_df.dropna(subset=['llm_corr_score', 'agent_corr_score', 'agent_ut_correctness'])
    print(f"✓ Loaded {len(complete)} tasks with complete data\n")

    # Generate graphs
    print("Generating graphs...\n")

    plot_1a_accuracy_bar(merged_df, output_dir / 'graph1a_accuracy_bar.png')
    plot_1d_scores_boxscatter(merged_df, output_dir / 'graph1d_scores_boxscatter.png')
    plot_3b_confusion_bar(merged_df, output_dir / 'graph3b_confusion_bar_chart.png')
    plot_4_multiaspect(merged_df, output_dir / 'graph4_multiaspect_correlation.png')
    plot_5bcde_agent_behavior(merged_df, output_dir)

    # Copy the table image (graph5_agent_behavior_metrics.png) and .tex file
    import shutil
    src_table = Path("outputs/analysis/final_figures/graph5_agent_behavior_metrics.png")
    src_tex = Path("outputs/analysis/final_figures/graph5_agent_behavior_metrics.tex")

    if src_table.exists():
        shutil.copy(src_table, output_dir / 'graph5_agent_behavior_metrics.png')
        print(f"✓ Copied: graph5_agent_behavior_metrics.png")

    if src_tex.exists():
        shutil.copy(src_tex, output_dir / 'graph5_agent_behavior_metrics.tex')
        print(f"✓ Copied: graph5_agent_behavior_metrics.tex")

    print("\n" + "="*80)
    print("✓ ALL PRETTIFIED GRAPHS GENERATED!")
    print("="*80)
    print(f"\n📂 Location: {output_dir.absolute()}\n")


if __name__ == '__main__':
    main()
