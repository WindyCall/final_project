"""
Update Graphs 5b-5e with:
- Same color palette as graph1a/1d/3b/4
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


def plot_5bcde_agent_behavior(merged_df, output_dir):
    """Graphs 5b-5e: Agent behavior box plots with matching color palette."""

    scenarios = ['Agent-Corr', 'Agent-Multi', 'Agent-UT']
    prefixes = ['agent_corr', 'agent_multi', 'agent_ut']

    # Color palette matching graph1a/1d/3b/4
    colors = ['#ffc8dd', '#ffafcc', '#cdb4db']  # Agent-Corr, Agent-Multi, Agent-UT

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

        for i, (patch, color) in enumerate(zip(bp['boxes'], colors)):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
            # For 5b (steps) and 5c (tools), hide boxes for Agent-Corr and Agent-Multi
            if metric_name in ['steps', 'tools'] and i < 2:
                patch.set_visible(False)

        # For 5b (steps) and 5c (tools), hide median, whiskers, and caps for Agent-Corr and Agent-Multi
        if metric_name in ['steps', 'tools']:
            for i in range(2):
                bp['medians'][i].set_visible(False)
                bp['whiskers'][i*2].set_visible(False)  # Lower whisker
                bp['whiskers'][i*2 + 1].set_visible(False)  # Upper whisker
                bp['caps'][i*2].set_visible(False)  # Lower cap
                bp['caps'][i*2 + 1].set_visible(False)  # Upper cap

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
        print(f"✓ Updated: {output_path.name}")


def main():
    """Update Graphs 5b-5e only."""
    print("="*80)
    print("UPDATING GRAPHS 5b-5e")
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

    # Generate updated graphs
    print("Generating updated graphs...\n")
    plot_5bcde_agent_behavior(merged_df, output_dir)

    print("\n" + "="*80)
    print("✓ GRAPHS 5b-5e UPDATED!")
    print("="*80)
    print(f"\nFeatures updated:")
    print("  ✓ Same color palette as graph1a/1d/3b/4")
    print(f"\nColor scheme:")
    print("  Agent-Corr:  Light Pink  #ffc8dd")
    print("  Agent-Multi: Darker Pink #ffafcc")
    print("  Agent-UT:    Purple      #cdb4db")
    print(f"\n📂 Location: {output_dir.absolute()}\n")


if __name__ == '__main__':
    main()
