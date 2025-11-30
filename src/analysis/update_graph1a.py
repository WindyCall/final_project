"""
Update Graph 1a with new color scheme and improved x-axis labels.

Color scheme:
- LLM judges: Green (Corr=light, Multi=darker)
- Agent judges: Blue (Corr=light, Multi=darker, UT=darkest)
"""

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from scipy import stats

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from analysis.data_loader import DataLoader


def wilson_score_interval(successes, n, confidence=0.95):
    """Calculate Wilson score confidence interval."""
    z = stats.norm.ppf(1 - (1 - confidence) / 2)
    p = successes / n
    denominator = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denominator
    margin = z * np.sqrt((p * (1 - p) + z**2 / (4 * n)) / n) / denominator
    return centre - margin, centre + margin


def plot_1a_accuracy_bar(merged_df, output_path):
    """Graph 1a: Accuracy bar chart with LLM=Green, Agent=Blue."""

    judges = ['LLM-Corr', 'LLM-Multi', 'Agent-Corr', 'Agent-Multi', 'Agent-UT']
    pred_cols = ['llm_corr_pred', 'llm_multi_pred', 'agent_corr_pred',
                 'agent_multi_pred', 'agent_ut_pred']

    # Color scheme: LLM=Green, Agent=Blue
    # Correctness=light, Multi=darker, UT=darkest
    colors = [
        '#bde0fe',  # LLM-Corr: Light green
        '#a2d2ff',  # LLM-Multi: Darker green
        '#ffc8dd',  # Agent-Corr: Light blue
        '#ffafcc',  # Agent-Multi: Darker blue
        '#cdb4db'   # Agent-UT: Dark blue
    ]

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
                  alpha=0.85, error_kw=dict(linewidth=1.5, capthick=1.5))

    # Labels
    ax.set_ylabel('Accuracy (%)', fontsize=13, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(judges, fontsize=9.5, rotation=10, ha='right')  # Smaller font + slight rotation
    ax.set_ylim(60, 105)
    ax.tick_params(axis='y', labelsize=11)

    # Clean styling
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='y', alpha=0.3, linestyle='--')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Updated: {output_path.name}")


def main():
    """Update Graph 1a only."""
    print("="*80)
    print("UPDATING GRAPH 1a")
    print("="*80)

    # Output directory
    output_dir = Path("outputs/analysis/final_figures_prettified")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    print("\nLoading data...")
    loader = DataLoader()
    merged_df = loader.merge_all()
    complete = merged_df.dropna(subset=['llm_corr_score', 'agent_corr_score'])
    print(f"✓ Loaded {len(complete)} tasks with complete data\n")

    # Generate updated graph
    print("Generating updated graph...\n")
    plot_1a_accuracy_bar(merged_df, output_dir / 'graph1a_accuracy_bar.png')

    print("\n" + "="*80)
    print("✓ GRAPH 1a UPDATED!")
    print("="*80)
    print(f"\nColor scheme applied:")
    print("  LLM-Corr:    Light Green  #A8D5A8")
    print("  LLM-Multi:   Darker Green #66BB6A")
    print("  Agent-Corr:  Light Blue   #90CAF9")
    print("  Agent-Multi: Darker Blue  #42A5F5")
    print("  Agent-UT:    Dark Blue    #1976D2")
    print(f"\nX-axis: Font size 9.5pt, rotation 10°")
    print(f"\n📂 Location: {output_dir.absolute()}\n")


if __name__ == '__main__':
    main()
