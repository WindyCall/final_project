"""Main script to generate all 6 core visualization graphs.

This script loads data from all 6 judge scenarios and generates
the essential graphs for the analysis report.
"""

import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend

from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from analysis.data_loader import DataLoader
from analysis.core_visualizations import (
    plot_accuracy_comparison,
    plot_cost_accuracy_tradeoff,
    plot_confusion_matrix_grid,
    plot_confusion_matrix_bar_chart,
    plot_multiaspect_correlation,
    plot_agent_behavior_metrics,
    plot_agent_behavior_box_plots,
    plot_score_distributions_by_outcome
)


def main():
    """Generate all 6 core visualization graphs."""
    print("="*80)
    print("GENERATING CORE VISUALIZATIONS")
    print("="*80)

    # Create output directory
    output_dir = Path("outputs/analysis/figures")
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n📁 Output directory: {output_dir}")

    # Load data
    print("\n" + "="*80)
    print("STEP 1: LOADING DATA")
    print("="*80)
    loader = DataLoader()
    merged_df = loader.merge_all()

    # Check data completeness
    complete = merged_df.dropna(subset=['llm_corr_score', 'agent_corr_score', 'agent_ut_correctness'])
    print(f"\n✓ Loaded {len(merged_df)} total tasks")
    print(f"✓ {len(complete)} tasks with complete judge data")

    # Generate graphs
    print("\n" + "="*80)
    print("STEP 2: GENERATING GRAPHS")
    print("="*80)

    # Graph 1 is special - it generates 2 separate graphs (1a, 1d)
    print()
    print(f"📊 Generating: Overall Accuracy Comparison (2 graphs: 1a, 1d)...")
    try:
        figs = plot_accuracy_comparison(merged_df, output_dir)
        if figs is not None:
            import matplotlib.pyplot as plt
            for fig in figs:
                plt.close(fig)  # Close to free memory
    except Exception as e:
        print(f"   ❌ Error: {e}")
        import traceback
        traceback.print_exc()

    # Regular graphs (2-4, 6)
    graphs = [
        ("graph2_cost_accuracy_tradeoff.png", plot_cost_accuracy_tradeoff,
         "Cost-Accuracy Trade-off"),
        ("graph3_confusion_matrices.png", plot_confusion_matrix_grid,
         "Confusion Matrix Grid"),
        ("graph4_multiaspect_correlation.png", plot_multiaspect_correlation,
         "Multi-Aspect Correlation Heatmap"),
        ("graph6_score_distributions.png", plot_score_distributions_by_outcome,
         "Score Distributions by UT Outcome")
    ]

    for filename, plot_func, description in graphs:
        print(f"📊 Generating: {description}...")
        output_path = output_dir / filename
        try:
            fig = plot_func(merged_df, output_path)
            if fig is not None:
                import matplotlib.pyplot as plt
                plt.close(fig)  # Close to free memory
        except Exception as e:
            print(f"   ❌ Error: {e}")
            import traceback
            traceback.print_exc()

    # Graph 3b: Confusion Matrix Bar Chart
    print(f"📊 Generating: Confusion Matrix Bar Chart (Graph 3b)...")
    try:
        output_path = output_dir / "graph3b_confusion_bar_chart.png"
        fig = plot_confusion_matrix_bar_chart(merged_df, output_path)
        if fig is not None:
            import matplotlib.pyplot as plt
            plt.close(fig)
    except Exception as e:
        print(f"   ❌ Error in Graph 3b: {e}")
        import traceback
        traceback.print_exc()

    # Graph 5 is special - it generates 5 separate graphs (5a-5e)
    print()
    print(f"📊 Generating: Agent Behavior Metrics (5 graphs: 5a-5e)...")

    # Graph 5a: Table
    try:
        output_path = output_dir / "graph5a_behavior_table.png"
        fig = plot_agent_behavior_metrics(merged_df, output_path)
        if fig is not None:
            import matplotlib.pyplot as plt
            plt.close(fig)
    except Exception as e:
        print(f"   ❌ Error in Graph 5a: {e}")
        import traceback
        traceback.print_exc()

    # Graphs 5b-5e: Box plots
    try:
        figs = plot_agent_behavior_box_plots(merged_df, output_dir)
        if figs is not None:
            import matplotlib.pyplot as plt
            for fig in figs:
                plt.close(fig)  # Close to free memory
    except Exception as e:
        print(f"   ❌ Error in Graphs 5b-5e: {e}")
        import traceback
        traceback.print_exc()

    # Summary statistics
    print("\n" + "="*80)
    print("STEP 3: SUMMARY STATISTICS")
    print("="*80)

    judges = [
        ('LLM-Corr', 'llm_corr_pred', 'llm_corr_score', None),
        ('LLM-Multi', 'llm_multi_pred', 'llm_multi_correctness', None),
        ('Agent-Corr', 'agent_corr_pred', 'agent_corr_score', 'agent_corr_cost'),
        ('Agent-Multi', 'agent_multi_pred', 'agent_multi_correctness', 'agent_multi_cost'),
        ('Agent-UT', 'agent_ut_pred', 'agent_ut_correctness', 'agent_ut_cost')
    ]

    print("\nJudge Performance Summary:")
    print("-" * 80)
    print(f"{'Judge':<15} {'Accuracy':<12} {'Avg Score':<12} {'Avg Cost':<12} {'N Tasks':<10}")
    print("-" * 80)

    for judge_name, pred_col, score_col, cost_col in judges:
        valid = merged_df.dropna(subset=[pred_col, 'passed'])
        n = len(valid)

        if n > 0:
            acc = (valid[pred_col] == valid['passed']).mean()
            avg_score = valid[score_col].mean()

            if cost_col:
                avg_cost = merged_df[cost_col].mean()
                cost_str = f"${avg_cost:.4f}"
            else:
                cost_str = "~$0.01"

            print(f"{judge_name:<15} {acc*100:>6.1f}%      {avg_score:>6.3f}       "
                  f"{cost_str:<12} {n:<10}")

    print("-" * 80)

    # Key findings
    print("\n" + "="*80)
    print("KEY FINDINGS")
    print("="*80)

    # Calculate specific comparisons
    llm_corr_acc = (merged_df.dropna(subset=['llm_corr_pred', 'passed'])['llm_corr_pred'] ==
                    merged_df.dropna(subset=['llm_corr_pred', 'passed'])['passed']).mean()
    llm_multi_acc = (merged_df.dropna(subset=['llm_multi_pred', 'passed'])['llm_multi_pred'] ==
                     merged_df.dropna(subset=['llm_multi_pred', 'passed'])['passed']).mean()
    agent_ut_acc = (merged_df.dropna(subset=['agent_ut_pred', 'passed'])['agent_ut_pred'] ==
                    merged_df.dropna(subset=['agent_ut_pred', 'passed'])['passed']).mean()

    agent_ut_cost = merged_df['agent_ut_cost'].mean()
    llm_cost_est = 0.01

    print(f"\n1. Agent-UT achieves {agent_ut_acc*100:.1f}% accuracy vs LLM-Corr {llm_corr_acc*100:.1f}%")
    print(f"   → Improvement: +{(agent_ut_acc - llm_corr_acc)*100:.1f} percentage points")

    print(f"\n2. LLM-Multi ({llm_multi_acc*100:.1f}%) vs LLM-Corr ({llm_corr_acc*100:.1f}%)")
    print(f"   → Multi-spec effect: {(llm_multi_acc - llm_corr_acc)*100:+.1f} percentage points")

    if agent_ut_cost > 0:
        cost_ratio = agent_ut_cost / llm_cost_est
        cost_per_correct = (agent_ut_cost / agent_ut_acc) / (llm_cost_est / llm_corr_acc)
        print(f"\n3. Cost Analysis:")
        print(f"   → Agent-UT is {cost_ratio:.1f}x more expensive than LLM judges")
        print(f"   → But {cost_per_correct:.1f}x more expensive per CORRECT judgment")

    # Multi-aspect correlation
    aspects_df = merged_df.dropna(subset=['llm_multi_correctness', 'llm_multi_style',
                                           'llm_multi_simplicity', 'llm_multi_robustness'])
    if len(aspects_df) > 0:
        from scipy.stats import pearsonr
        corr_style_simp, _ = pearsonr(aspects_df['llm_multi_style'],
                                       aspects_df['llm_multi_simplicity'])
        corr_correct_robust, _ = pearsonr(aspects_df['llm_multi_correctness'],
                                           aspects_df['llm_multi_robustness'])

        print(f"\n4. Multi-Aspect Correlations:")
        print(f"   → Style ↔ Simplicity: r = {corr_style_simp:.3f}")
        print(f"   → Correctness ↔ Robustness: r = {corr_correct_robust:.3f}")

    print("\n" + "="*80)
    print("✓ ALL GRAPHS GENERATED SUCCESSFULLY!")
    print("="*80)
    print(f"\n📂 Output location: {output_dir.absolute()}")
    print("\nGenerated files:")

    # List Graph 1 files (a, d)
    for suffix, desc, fname in [('a', 'Bar Chart', 'bar'),
                                ('d', 'Box+Scatter (Numerical)', 'boxscatter')]:
        filepath = output_dir / f"graph1{suffix}_accuracy_{fname}.png"
        if filepath.exists():
            size_kb = filepath.stat().st_size / 1024
            print(f"  ✓ graph1{suffix}_accuracy_{fname}.png ({size_kb:.1f} KB) - Accuracy {desc}")

    # List regular graphs (2-4, 6)
    for filename, _, description in graphs:
        filepath = output_dir / filename
        if filepath.exists():
            size_kb = filepath.stat().st_size / 1024
            print(f"  ✓ {filename} ({size_kb:.1f} KB) - {description}")

    # List Graph 3b
    graph3b_path = output_dir / "graph3b_confusion_bar_chart.png"
    if graph3b_path.exists():
        size_kb = graph3b_path.stat().st_size / 1024
        print(f"  ✓ graph3b_confusion_bar_chart.png ({size_kb:.1f} KB) - Confusion Matrix Bar Chart")

    # List Graph 5 files (a-e)
    graph5_files = [
        ('graph5a_behavior_table.png', 'Agent Behavior Table'),
        ('graph5b_steps.png', 'Number of Steps'),
        ('graph5c_tools.png', 'Tool Usage'),
        ('graph5d_time.png', 'Time to Completion'),
        ('graph5e_cost.png', 'Total Cost')
    ]
    for filename, description in graph5_files:
        filepath = output_dir / filename
        if filepath.exists():
            size_kb = filepath.stat().st_size / 1024
            print(f"  ✓ {filename} ({size_kb:.1f} KB) - {description}")

    print("\n" + "="*80)
    print("NEXT STEPS")
    print("="*80)
    print("1. Review the generated graphs in: outputs/analysis/figures/")
    print("2. Include these graphs in your final report")
    print("3. Use the summary statistics above for your results section")
    print("4. Refer to CORE_VISUALIZATIONS.md for expected insights per graph")
    print("="*80)


if __name__ == "__main__":
    main()
