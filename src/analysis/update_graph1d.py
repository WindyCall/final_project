"""
Update Graph 1d with:
- Same color scheme as graph1a (LLM=green, Agent=blue)
- Mean markers (blue diamonds)
- Same x-axis format (9.5pt, 10° rotation)
"""

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from analysis.data_loader import DataLoader


def plot_1d_scores_boxscatter(merged_df, output_path):
    """Graph 1d: Numerical scores scatter only with mean markers (no boxplot)."""

    judges = ['Ground Truth', 'LLM-Corr', 'LLM-Multi', 'Agent-Corr', 'Agent-Multi', 'Agent-UT']
    score_cols = ['pass_rate', 'llm_corr_score', 'llm_multi_correctness', 'agent_corr_score',
                  'agent_multi_correctness', 'agent_ut_correctness']

    # Color scheme: Ground Truth (gold) + LLM/Agent colors from graph1a
    colors = [
        '#FFD700',  # Ground Truth: Gold
        '#bde0fe',  # LLM-Corr: Light blue
        '#a2d2ff',  # LLM-Multi: Darker blue
        '#ffc8dd',  # Agent-Corr: Light pink
        '#ffafcc',  # Agent-Multi: Darker pink
        '#cdb4db'   # Agent-UT: Purple
    ]

    all_scores = []
    for score_col in score_cols:
        valid = merged_df.dropna(subset=[score_col])
        all_scores.append(valid[score_col].values)

    # Compact figure - wider to accommodate 6 judges
    fig, ax = plt.subplots(figsize=(5.5, 3))

    x_pos = np.arange(len(judges))

    # Scatter points only (no boxplot)
    for i, (scores, color) in enumerate(zip(all_scores, colors)):
        x_jitter = np.random.normal(i, 0.08, size=len(scores))
        ax.scatter(x_jitter, scores, alpha=0.7, s=25, color=color,
                  edgecolors='gray', linewidths=0.3, zorder=5)

    # Add mean markers (dark purple diamonds)
    for i, scores in enumerate(all_scores):
        mean_score = scores.mean()
        ax.scatter(i, mean_score, s=50, color='#6A1B9A', marker='D',
                  edgecolors='black', linewidths=0.8, zorder=10,
                  label='Mean' if i == 0 else '')

    # Labels - same format as graph1a
    ax.set_ylabel('Score', fontsize=13, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(judges, fontsize=9.5, rotation=10, ha='right')  # Same as graph1a
    ax.set_ylim(-0.05, 1.05)
    ax.tick_params(axis='y', labelsize=11)

    # Clean styling
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='y', alpha=0.3, linestyle='--')

    # Add legend for mean
    ax.legend(loc='lower right', fontsize=9, framealpha=0.9)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Updated: {output_path.name}")


def main():
    """Update Graph 1d only."""
    print("="*80)
    print("UPDATING GRAPH 1d")
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
    plot_1d_scores_boxscatter(merged_df, output_dir / 'graph1d_scores_boxscatter.png')

    print("\n" + "="*80)
    print("✓ GRAPH 1d UPDATED!")
    print("="*80)
    print(f"\nFeatures added:")
    print("  ✓ Same color scheme as Graph 1a (LLM=green, Agent=blue)")
    print("  ✓ Mean markers (blue diamonds)")
    print("  ✓ Same x-axis format (9.5pt font, 10° rotation)")
    print("  ✓ Legend showing mean")
    print(f"\n📂 Location: {output_dir.absolute()}\n")


if __name__ == '__main__':
    main()
