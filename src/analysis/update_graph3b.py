"""
Update Graph 3b with:
- Same color palette as graph1a/1d
- Legend at top right
"""

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from sklearn.metrics import confusion_matrix

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from analysis.data_loader import DataLoader


def plot_3b_confusion_bar(merged_df, output_path):
    """Graph 3b: Confusion matrix bar chart with matching colors."""

    judges = ['LLM-Corr', 'LLM-Multi', 'Agent-Corr', 'Agent-Multi', 'Agent-UT']
    pred_cols = ['llm_corr_pred', 'llm_multi_pred', 'agent_corr_pred',
                 'agent_multi_pred', 'agent_ut_pred']

    # Same color palette as graph1a/1d
    colors = [
        '#bde0fe',  # LLM-Corr: Light blue
        '#a2d2ff',  # LLM-Multi: Darker blue
        '#ffc8dd',  # Agent-Corr: Light pink
        '#ffafcc',  # Agent-Multi: Darker pink
        '#cdb4db'   # Agent-UT: Purple
    ]

    # Reorder: True categories on top, False categories below (reversed for barh)
    categories = ['False Negative', 'False Positive', 'True Negative', 'True Positive']
    # Y-axis labels with line breaks
    category_labels = ['False\nNegative', 'False\nPositive', 'True\nNegative', 'True\nPositive']
    # Keep original order for data collection
    orig_categories = ['True Positive', 'True Negative', 'False Positive', 'False Negative']
    data = {cat: [] for cat in orig_categories}

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

    # Compact figure - taller for better readability
    fig, ax = plt.subplots(figsize=(4, 3.8))

    bar_height = 0.15
    y_positions = np.arange(len(categories))

    # Store bar objects for adding percentage labels
    bars_list = []
    for i, (judge, color) in enumerate(zip(judges, colors)):
        # Map reordered categories to original data
        values = [data[cat][i] for cat in categories]
        # Reverse offset: LLMs first (above), then Agents (below)
        offset = (2 - i) * bar_height
        bars = ax.barh(y_positions + offset, values, bar_height,
                       label=judge, color=color, edgecolor='black', linewidth=0.8,
                       alpha=0.85)
        bars_list.append((bars, values, y_positions + offset))

    # Add percentage labels to the right of each bar
    for bars, values, y_pos in bars_list:
        for bar, value, y in zip(bars, values, y_pos):
            if value > 2:  # Only show label if bar is wide enough
                ax.text(value + 1.5, y, f'{value:.1f}%',
                       va='center', ha='left', fontsize=7, color='black')

    ax.set_yticks(y_positions)
    ax.set_yticklabels(category_labels, fontsize=10, fontweight='bold')  # With line breaks
    ax.set_xlabel('Percentage (%)', fontsize=11, fontweight='bold')  # Further reduced
    ax.set_xlim(0, 85)  # Range 0-85
    ax.tick_params(axis='both', labelsize=9)  # Further reduced
    ax.grid(axis='x', alpha=0.3, linestyle='--')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Legend at bottom right - vertical (one mode per line)
    ax.legend(loc='lower right', fontsize=8, framealpha=0.9, ncol=1)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Updated: {output_path.name}")


def main():
    """Update Graph 3b only."""
    print("="*80)
    print("UPDATING GRAPH 3b")
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
    plot_3b_confusion_bar(merged_df, output_dir / 'graph3b_confusion_bar_chart.png')

    print("\n" + "="*80)
    print("✓ GRAPH 3b UPDATED!")
    print("="*80)
    print(f"\nFeatures updated:")
    print("  ✓ Same color palette as graph1a/1d")
    print("  ✓ Legend moved to top right")
    print(f"\nColor scheme:")
    print("  LLM-Corr:    Light Blue  #bde0fe")
    print("  LLM-Multi:   Darker Blue #a2d2ff")
    print("  Agent-Corr:  Light Pink  #ffc8dd")
    print("  Agent-Multi: Darker Pink #ffafcc")
    print("  Agent-UT:    Purple      #cdb4db")
    print(f"\n📂 Location: {output_dir.absolute()}\n")


if __name__ == '__main__':
    main()
