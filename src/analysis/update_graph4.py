"""
Update Graph 4 with:
- Same color palette as graph1a/1d/3b
- X-axis rotation 20° instead of 45°, font size 8
- Correctness subplot 1.5x wider than other subplots
"""

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np
from pathlib import Path
from scipy import stats

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from analysis.data_loader import DataLoader


def plot_4_multiaspect(merged_df, output_path):
    """Graph 4: Multi-aspect correlation with matching color palette."""

    aspects = ['Correctness', 'Style', 'Simplicity', 'Robustness']

    # Color palette matching graph1a/1d/3b
    judge_configs = {
        'Correctness': [
            ('Ground Truth', 'pass_rate', '#FFD700'),  # Gold
            ('LLM-Corr', 'llm_corr_score', '#bde0fe'),  # Light blue
            ('LLM-Multi', 'llm_multi_correctness', '#a2d2ff'),  # Darker blue
            ('Agent-Corr', 'agent_corr_score', '#ffc8dd'),  # Light pink
            ('Agent-Multi', 'agent_multi_correctness', '#ffafcc'),  # Darker pink
            ('Agent-UT', 'agent_ut_correctness', '#cdb4db')  # Purple
        ],
        'Style': [
            ('LLM-Multi', 'llm_multi_style', '#a2d2ff'),
            ('Agent-Multi', 'agent_multi_style', '#ffafcc'),
            ('Agent-UT', 'agent_ut_style', '#cdb4db')
        ],
        'Simplicity': [
            ('LLM-Multi', 'llm_multi_simplicity', '#a2d2ff'),
            ('Agent-Multi', 'agent_multi_simplicity', '#ffafcc'),
            ('Agent-UT', 'agent_ut_simplicity', '#cdb4db')
        ],
        'Robustness': [
            ('LLM-Multi', 'llm_multi_robustness', '#a2d2ff'),
            ('Agent-Multi', 'agent_multi_robustness', '#ffafcc'),
            ('Agent-UT', 'agent_ut_robustness', '#cdb4db')
        ]
    }

    # Create figure with GridSpec - Correctness subplot 1.5x wider
    fig = plt.figure(figsize=(11, 2.5))
    gs = GridSpec(1, 4, figure=fig, width_ratios=[1.5, 1, 1, 1], wspace=0.3)

    axes = [fig.add_subplot(gs[0, i]) for i in range(4)]

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
                      alpha=0.85, error_kw=dict(linewidth=1.2, capthick=1.5))

        # Labels with 20° rotation
        ax.set_ylabel('Score' if idx == 0 else '', fontsize=11, fontweight='bold')
        ax.set_title(aspect, fontsize=12, fontweight='bold', pad=8)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(judges, rotation=20, ha='right', fontsize=8)
        ax.set_ylim(0, 1.15)
        ax.tick_params(axis='both', labelsize=9)
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Updated: {output_path.name}")


def main():
    """Update Graph 4 only."""
    print("="*80)
    print("UPDATING GRAPH 4")
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
    plot_4_multiaspect(merged_df, output_dir / 'graph4_multiaspect_correlation.png')

    print("\n" + "="*80)
    print("✓ GRAPH 4 UPDATED!")
    print("="*80)
    print(f"\nFeatures updated:")
    print("  ✓ Same color palette as graph1a/1d/3b")
    print("  ✓ X-axis rotation changed to 20° (from 45°), font size 8")
    print("  ✓ Correctness subplot 1.5x wider than other subplots")
    print(f"\nColor scheme:")
    print("  Ground Truth: Gold      #FFD700")
    print("  LLM-Corr:     Light Blue  #bde0fe")
    print("  LLM-Multi:    Darker Blue #a2d2ff")
    print("  Agent-Corr:   Light Pink  #ffc8dd")
    print("  Agent-Multi:  Darker Pink #ffafcc")
    print("  Agent-UT:     Purple      #cdb4db")
    print(f"\n📂 Location: {output_dir.absolute()}\n")


if __name__ == '__main__':
    main()
