# Core Visualizations for AI Judges vs. Unit Tests Analysis
## 6 Essential Graphs with Expected Insights

---

## Graph 1: Overall Accuracy Comparison (Bar Chart with Error Bars)

### **Purpose**
Answers **RQ1** (Terminal access impact) and **RQ3** (Self-testing value)

### **Design**
```
X-axis: Judge configuration (6 bars)
  - Ground Truth (100% baseline)
  - LLM-Corr
  - LLM-Multi
  - Agent-Corr
  - Agent-Multi
  - Agent-UT

Y-axis: Binary accuracy (0-100%)

Features:
  - 95% confidence intervals (error bars)
  - Color coding:
    * Gold: Ground Truth
    * Blue shades: LLM judges
    * Green shades: Agent judges
  - Dashed horizontal line at 100% (perfect accuracy)
  - Value labels on top of each bar
```

### **Expected Insights**

**Hypothesis A: Agent-UT dominates**
```
Expected pattern: Agent-UT (97%) >> LLM-Multi (77%) > LLM-Corr (73%)
                  Agent-UT (97%) >> Agent-Multi (~75-80%) > Agent-Corr (~75-80%)

If TRUE → Shows that self-testing dramatically improves accuracy
Implication: When accuracy is critical, Agent-UT is worth the cost
```

**Hypothesis B: Terminal access provides marginal gains**
```
Expected pattern: Agent-Corr > LLM-Corr by ~5-7%
                  Agent-Multi > LLM-Multi by ~0-5%

If TRUE → Terminal access helps, but not dramatically (without self-testing)
If FALSE (no difference) → Static LLM analysis is as good as agent execution
Implication: For correctness-only tasks, LLMs may suffice if cost-sensitive
```

**Hypothesis C: Multi-spec prompting doesn't hurt correctness**
```
Expected pattern: LLM-Multi ≈ LLM-Corr (or slightly better)
                  Agent-Multi ≈ Agent-Corr

If TRUE → Safe to ask for multiple dimensions without sacrificing accuracy
If FALSE (Multi worse) → Multi-spec prompting distracts from correctness
Implication: Can get Style/Simplicity/Robustness feedback "for free"
```

### **Implementation Code**
```python
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import binom

def wilson_ci(successes, n, confidence=0.95):
    """Calculate Wilson score confidence interval."""
    from scipy.stats import norm
    z = norm.ppf(1 - (1-confidence)/2)
    p_hat = successes / n
    denominator = 1 + z**2/n
    center = (p_hat + z**2/(2*n)) / denominator
    margin = z * np.sqrt(p_hat*(1-p_hat)/n + z**2/(4*n**2)) / denominator
    return center - margin, center + margin

def plot_accuracy_comparison(accuracies_df):
    """
    accuracies_df columns: ['judge', 'accuracy', 'n_tasks']
    """
    fig, ax = plt.subplots(figsize=(12, 7))

    # Calculate confidence intervals
    ci_lower = []
    ci_upper = []
    for _, row in accuracies_df.iterrows():
        n = row['n_tasks']
        successes = int(row['accuracy'] * n)
        lower, upper = wilson_ci(successes, n)
        ci_lower.append(lower)
        ci_upper.append(upper)

    accuracies_df['ci_lower'] = ci_lower
    accuracies_df['ci_upper'] = ci_upper

    # Define colors
    colors = {
        'Ground Truth': '#FFD700',  # Gold
        'LLM-Corr': '#4A90E2',      # Light blue
        'LLM-Multi': '#2E5C8A',     # Dark blue
        'Agent-Corr': '#7FBA7A',    # Light green
        'Agent-Multi': '#4A9B4D',   # Medium green
        'Agent-UT': '#2D6B2F'       # Dark green
    }

    # Plot bars
    x_pos = np.arange(len(accuracies_df))
    bars = ax.bar(
        x_pos,
        accuracies_df['accuracy'],
        yerr=[
            accuracies_df['accuracy'] - accuracies_df['ci_lower'],
            accuracies_df['ci_upper'] - accuracies_df['accuracy']
        ],
        capsize=8,
        color=[colors[j] for j in accuracies_df['judge']],
        edgecolor='black',
        linewidth=2,
        alpha=0.8
    )

    # Styling
    ax.set_ylabel('Binary Accuracy', fontsize=14, fontweight='bold')
    ax.set_xlabel('Judge Configuration', fontsize=14, fontweight='bold')
    ax.set_title('Accuracy Comparison: Which Judge Best Matches Unit Tests?',
                 fontsize=16, fontweight='bold', pad=20)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(accuracies_df['judge'], rotation=45, ha='right')
    ax.set_ylim(0, 1.1)

    # Reference line
    ax.axhline(y=1.0, color='red', linestyle='--', linewidth=2,
               label='Perfect Accuracy (100%)', alpha=0.7, zorder=0)

    # Add value labels
    for i, (bar, acc) in enumerate(zip(bars, accuracies_df['accuracy'])):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'{acc*100:.1f}%',
                ha='center', va='bottom', fontweight='bold', fontsize=11)

    # Grid and legend
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.legend(fontsize=11)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    return fig

# Expected output:
# Ground Truth: 100% (baseline)
# LLM-Corr: 72.9% ± 5%
# LLM-Multi: 77.1% ± 5%
# Agent-Corr: ~75-80% ± 5%
# Agent-Multi: ~75-80% ± 5%
# Agent-UT: 97.1% ± 2%
```

---

## Graph 2: Cost-Accuracy Trade-off (Scatter Plot with Pareto Frontier)

### **Purpose**
Answers **RQ4** (Cost-benefit analysis)

### **Design**
```
X-axis: Average cost per task (USD, log scale)
Y-axis: Binary accuracy (%)

Points: 6 judge configurations
  - Size: Proportional to throughput (tasks/hour)
  - Color: Same as Graph 1
  - Labels: Judge names

Features:
  - Pareto frontier line (connecting efficient judges)
  - Quadrant lines (median cost, median accuracy)
  - Annotations: Cost per 100 tasks, Tasks/hour
```

### **Expected Insights**

**Hypothesis A: Agent-UT is most expensive but most accurate**
```
Expected pattern:
  Agent-UT: High cost (~$0.30), High accuracy (97%)
  Agent-Multi: Medium cost (~$0.10), Medium accuracy (~77%)
  LLM judges: Low cost (~$0.01), Medium accuracy (~73-77%)

If TRUE → Clear cost-accuracy spectrum exists
Implication: Choose judge based on accuracy requirements vs. budget
```

**Hypothesis B: LLM-Multi is the "sweet spot"**
```
Expected pattern: LLM-Multi achieves 77% accuracy at $0.01/task
                  Agent-Multi achieves similar accuracy at 10x cost

If TRUE → LLM-Multi is most cost-effective for moderate accuracy needs
If FALSE (Agent-Multi much better) → Agent autonomy worth the cost
Implication: For most applications, LLM-Multi offers best value
```

**Hypothesis C: Agent-UT is only worth it for high-stakes applications**
```
Expected calculation:
  Agent-UT: $0.30 / 0.97 accuracy = $0.31 per correct judgment
  LLM-Multi: $0.01 / 0.77 accuracy = $0.013 per correct judgment

  Cost ratio: 24x more expensive per correct judgment

If TRUE → Agent-UT justified only when accuracy >> cost
Implication: Use for critical code, high-value projects, safety-critical systems
```

**Hypothesis D: Throughput inversely correlates with accuracy**
```
Expected pattern:
  LLM judges: ~720 tasks/hour (5s each)
  Agent-UT: ~12 tasks/hour (300s each)

If TRUE → Speed vs. accuracy tradeoff exists
Implication: LLMs for batch processing, Agents for interactive/high-value tasks
```

### **Implementation Code**
```python
def plot_cost_accuracy_tradeoff(metrics_df):
    """
    metrics_df columns: ['judge', 'accuracy', 'cost_per_task', 'time_per_task']
    """
    fig, ax = plt.subplots(figsize=(12, 8))

    # Calculate throughput
    metrics_df['tasks_per_hour'] = 3600 / metrics_df['time_per_task']
    metrics_df['cost_per_100'] = metrics_df['cost_per_task'] * 100

    # Define colors (same as Graph 1)
    colors = {
        'LLM-Corr': '#4A90E2', 'LLM-Multi': '#2E5C8A',
        'Agent-Corr': '#7FBA7A', 'Agent-Multi': '#4A9B4D', 'Agent-UT': '#2D6B2F'
    }

    # Scatter plot
    for _, row in metrics_df.iterrows():
        ax.scatter(
            row['cost_per_task'],
            row['accuracy'] * 100,
            s=row['tasks_per_hour'] * 10,  # Size proportional to throughput
            color=colors[row['judge']],
            alpha=0.7,
            edgecolor='black',
            linewidth=2,
            label=row['judge']
        )

        # Annotate
        ax.annotate(
            f"{row['judge']}\n${row['cost_per_100']:.2f}/100 tasks\n{row['tasks_per_hour']:.0f} tasks/hr",
            xy=(row['cost_per_task'], row['accuracy'] * 100),
            xytext=(10, 10),
            textcoords='offset points',
            fontsize=9,
            bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8)
        )

    # Pareto frontier (optional: connect efficient points)
    # Sort by cost and connect points where accuracy increases
    sorted_df = metrics_df.sort_values('cost_per_task')
    pareto_points = []
    max_acc = 0
    for _, row in sorted_df.iterrows():
        if row['accuracy'] > max_acc:
            pareto_points.append((row['cost_per_task'], row['accuracy'] * 100))
            max_acc = row['accuracy']

    if len(pareto_points) > 1:
        pareto_x, pareto_y = zip(*pareto_points)
        ax.plot(pareto_x, pareto_y, 'k--', alpha=0.5, linewidth=2,
                label='Pareto Frontier', zorder=0)

    # Quadrant lines
    median_cost = metrics_df['cost_per_task'].median()
    median_acc = metrics_df['accuracy'].median() * 100
    ax.axvline(median_cost, color='gray', linestyle=':', alpha=0.5)
    ax.axhline(median_acc, color='gray', linestyle=':', alpha=0.5)

    # Styling
    ax.set_xlabel('Cost per Task (USD)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Accuracy (%)', fontsize=14, fontweight='bold')
    ax.set_title('Cost-Accuracy Trade-off: Is Agent-UT Worth the Price?',
                 fontsize=16, fontweight='bold', pad=20)
    ax.set_xscale('log')
    ax.set_xlim(0.005, 0.5)
    ax.set_ylim(60, 105)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10, loc='lower right')

    plt.tight_layout()
    return fig

# Expected output:
# LLM-Corr: ($0.01, 73%) - Large circle (fast)
# LLM-Multi: ($0.01, 77%) - Large circle (fast)
# Agent-Corr: ($0.05, 77%) - Medium circle
# Agent-Multi: ($0.10, 77%) - Small circle
# Agent-UT: ($0.30, 97%) - Tiny circle (slow) but highest accuracy
```

---

## Graph 3: Confusion Matrix Grid (2×3 Heatmaps)

### **Purpose**
Answers **RQ1** (Terminal access) and **RQ6** (Error patterns)

### **Design**
```
Layout: 2×3 subplot grid (6 judges, excluding Ground Truth)

Each subplot: 2×2 confusion matrix
Rows: Judge prediction (PASS/FAIL)
Cols: Unit test outcome (PASS/FAIL)

Cells: TP, FP, FN, TN counts
Color: Normalized by total (darker = more cases)
Annotations: Raw counts + percentages

Title: Judge name + Overall accuracy + F1 score
```

### **Expected Insights**

**Hypothesis A: Agent-UT has highest TP rate, lowest FP rate**
```
Expected pattern:
  Agent-UT: TP ~50, FP ~2, FN ~0, TN ~18
  LLM judges: TP ~38, FP ~5, FN ~14, TN ~13

If TRUE → Agent-UT rarely makes mistakes (both FP and FN low)
Implication: Trustworthy for production use
```

**Hypothesis B: LLMs have higher False Negative rate**
```
Expected pattern:
  LLM judges: More FN (judge says FAIL, but UT passed)
  Reason: Conservative, can't execute to verify

If TRUE → LLMs over-reject correct solutions
Implication: Use LLMs for screening, not final judgment
```

**Hypothesis C: Agents have lower False Positive rate**
```
Expected pattern:
  Agent judges: Fewer FP (judge says PASS, but UT failed)
  Reason: Can execute and catch runtime errors

If TRUE → Agents catch bugs LLMs miss
Implication: Execution capability is key advantage
```

**Hypothesis D: Multi-spec doesn't change TP/FP rates much**
```
Expected pattern:
  LLM-Corr vs LLM-Multi: Similar confusion matrices
  Agent-Corr vs Agent-Multi: Similar confusion matrices

If TRUE → Multi-spec orthogonal to correctness
If FALSE → Multi-spec helps/hurts correctness
Implication: Can ask for extra dimensions without risk
```

### **Implementation Code**
```python
from sklearn.metrics import confusion_matrix, classification_report

def plot_confusion_matrix_grid(predictions_df, ground_truth_df):
    """
    predictions_df: Multi-column DF with judge predictions (binary)
    ground_truth_df: UT outcomes (binary: 1=PASS, 0=FAIL)
    """
    judges = ['LLM-Corr', 'LLM-Multi', 'Agent-Corr', 'Agent-Multi', 'Agent-UT']

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.flatten()

    for idx, judge in enumerate(judges):
        ax = axes[idx]

        # Get predictions and ground truth
        y_true = ground_truth_df['passed']  # 1=PASS, 0=FAIL
        y_pred = predictions_df[judge]

        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        # cm[0, 0] = TN, cm[0, 1] = FP
        # cm[1, 0] = FN, cm[1, 1] = TP

        # Calculate metrics
        tn, fp, fn, tp = cm.ravel()
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        # Normalize for color scale
        cm_normalized = cm.astype('float') / cm.sum()

        # Plot heatmap
        im = ax.imshow(cm_normalized, cmap='Blues', aspect='auto', vmin=0, vmax=0.5)

        # Annotations
        for i in range(2):
            for j in range(2):
                count = cm[i, j]
                pct = cm_normalized[i, j] * 100
                text = f'{count}\n({pct:.1f}%)'
                color = 'white' if cm_normalized[i, j] > 0.25 else 'black'
                ax.text(j, i, text, ha='center', va='center',
                       color=color, fontsize=12, fontweight='bold')

        # Labels
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(['UT FAIL', 'UT PASS'], fontsize=10)
        ax.set_yticklabels(['Judge FAIL', 'Judge PASS'], fontsize=10)
        ax.set_xlabel('Ground Truth', fontsize=11, fontweight='bold')
        ax.set_ylabel('Judge Prediction', fontsize=11, fontweight='bold')

        # Title with metrics
        title = f'{judge}\nAcc={accuracy*100:.1f}% | F1={f1:.2f}'
        ax.set_title(title, fontsize=12, fontweight='bold', pad=10)

    # Hide unused subplot
    axes[5].axis('off')

    # Overall title
    fig.suptitle('Confusion Matrices: Where Do Judges Make Mistakes?',
                 fontsize=16, fontweight='bold', y=0.98)

    # Colorbar
    cbar = fig.colorbar(im, ax=axes, orientation='horizontal',
                        pad=0.05, shrink=0.8, aspect=30)
    cbar.set_label('Proportion of Total Cases', fontsize=11, fontweight='bold')

    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    return fig

# Expected output:
# Agent-UT: Very dark TP cell, very light FP/FN cells (best)
# LLM judges: Darker FN cells (over-reject correct code)
# Agent judges: Lighter FP cells (execution catches errors)
```

---

## Graph 4: Multi-Aspect Correlation Heatmap

### **Purpose**
Answers **RQ5** (Multi-aspect value beyond correctness)

### **Design**
```
5×5 correlation matrix heatmap

Variables:
  1. Correctness score (from multi-spec judges)
  2. Style score
  3. Simplicity score
  4. Robustness score
  5. Actual UT pass rate (ground truth)

Color: Pearson correlation coefficient (-1 to +1)
  - Red: Strong positive correlation
  - Blue: Strong negative correlation
  - White: No correlation

Annotations: Correlation values + significance stars
  * p<0.05, ** p<0.01, *** p<0.001
```

### **Expected Insights**

**Hypothesis A: Correctness moderately correlates with UT pass rate**
```
Expected: Correctness ↔ UT pass rate: R = 0.40-0.50

If TRUE → Multi-spec correctness scores somewhat predictive
If FALSE (R > 0.7) → Very predictive, comparable to binary judgment
If FALSE (R < 0.3) → Scores don't capture correctness well
Implication: Validates (or invalidates) continuous scoring approach
```

**Hypothesis B: Style/Simplicity/Robustness are orthogonal to correctness**
```
Expected:
  Style ↔ Correctness: R = 0.2-0.4 (weak correlation)
  Simplicity ↔ Correctness: R = 0.2-0.3
  Robustness ↔ Correctness: R = 0.3-0.5

If TRUE → These aspects provide independent value
Implication: Multi-spec captures quality dimensions beyond correctness
```

**Hypothesis C: Style and Simplicity are inversely correlated**
```
Expected: Style ↔ Simplicity: R = -0.3 to -0.5

Reasoning: Well-documented code (high Style) may be more verbose (low Simplicity)

If TRUE → Trade-off between documentation and conciseness
Implication: Different use cases prioritize differently (education vs. production)
```

**Hypothesis D: Robustness correlates with Correctness**
```
Expected: Robustness ↔ Correctness: R = 0.5-0.7

Reasoning: Robust code (edge cases, error handling) more likely to pass tests

If TRUE → Robustness score is useful correctness proxy
Implication: Robustness feedback helps improve correctness
```

### **Implementation Code**
```python
from scipy.stats import pearsonr
import seaborn as sns

def plot_multiaspect_correlation(multiaspect_df, ground_truth_df):
    """
    multiaspect_df: Columns ['task_id', 'correctness', 'style', 'simplicity', 'robustness']
    ground_truth_df: Columns ['task_id', 'pass_rate'] (0.0 to 1.0)
    """
    # Merge datasets
    merged = multiaspect_df.merge(ground_truth_df, on='task_id')

    # Select columns for correlation
    aspects = ['correctness', 'style', 'simplicity', 'robustness', 'pass_rate']
    labels = ['Correctness\n(Judge)', 'Style\n(Judge)', 'Simplicity\n(Judge)',
              'Robustness\n(Judge)', 'UT Pass Rate\n(Ground Truth)']

    # Calculate correlation matrix
    corr_matrix = merged[aspects].corr(method='pearson')

    # Calculate p-values
    n = len(merged)
    p_values = np.zeros((len(aspects), len(aspects)))
    for i in range(len(aspects)):
        for j in range(len(aspects)):
            if i != j:
                _, p_values[i, j] = pearsonr(merged[aspects[i]], merged[aspects[j]])

    # Create significance annotations
    annot = np.empty_like(corr_matrix, dtype=object)
    for i in range(len(aspects)):
        for j in range(len(aspects)):
            r = corr_matrix.iloc[i, j]
            p = p_values[i, j]
            if i == j:
                annot[i, j] = f'{r:.2f}'
            else:
                sig = ''
                if p < 0.001:
                    sig = '***'
                elif p < 0.01:
                    sig = '**'
                elif p < 0.05:
                    sig = '*'
                annot[i, j] = f'{r:.2f}{sig}'

    # Plot heatmap
    fig, ax = plt.subplots(figsize=(12, 10))

    sns.heatmap(
        corr_matrix,
        annot=annot,
        fmt='',
        cmap='RdBu_r',
        center=0,
        vmin=-1,
        vmax=1,
        square=True,
        linewidths=2,
        cbar_kws={'label': 'Pearson Correlation Coefficient', 'shrink': 0.8},
        ax=ax,
        annot_kws={'fontsize': 11, 'fontweight': 'bold'}
    )

    # Labels
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=11)
    ax.set_yticklabels(labels, rotation=0, fontsize=11)
    ax.set_title('Multi-Aspect Correlation: Do Style/Simplicity/Robustness Provide Value Beyond Correctness?',
                 fontsize=14, fontweight='bold', pad=20)

    # Add footnote
    fig.text(0.5, 0.02, 'Significance: * p<0.05, ** p<0.01, *** p<0.001',
             ha='center', fontsize=10, style='italic')

    plt.tight_layout(rect=[0, 0.03, 1, 1])
    return fig

# Expected output:
# Diagonal: 1.00 (perfect self-correlation)
# Correctness ↔ UT pass: 0.40-0.50*** (moderate, significant)
# Style ↔ Simplicity: -0.30 to -0.50** (inverse trade-off)
# Robustness ↔ Correctness: 0.50-0.70*** (strong, significant)
# Style/Simplicity ↔ UT pass: 0.10-0.30* (weak, orthogonal)
```

---

## Graph 5: Agent Behavior Metrics (Box Plot Grid)

### **Purpose**
Answers **RQ7** (Agent behavior patterns)

### **Design**
```
Layout: 2×2 grid of box plots

Subplots:
  1. Number of steps (trajectory length)
  2. Tool usage count (bash + edits)
  3. Time to completion (seconds)
  4. Total cost (USD)

Each subplot:
  X-axis: Agent scenario (Agent-Corr, Agent-Multi, Agent-UT)
  Y-axis: Metric value
  Box plot: Median, quartiles, outliers
  Overlay: Individual points (jittered, semi-transparent)
  Statistical annotation: Kruskal-Wallis p-value
```

### **Expected Insights**

**Hypothesis A: Agent-UT uses significantly more steps/tools**
```
Expected:
  Agent-Corr: ~10 steps, ~5 tool calls
  Agent-Multi: ~15 steps, ~7 tool calls
  Agent-UT: ~40 steps, ~20 tool calls (writes + runs tests)

If TRUE → Self-testing requires extensive autonomous work
Implication: Agent-UT's high cost justified by extra work done
```

**Hypothesis B: More steps → Higher accuracy (but diminishing returns)**
```
Expected correlation:
  Agent-Corr (10 steps) → 75% accuracy
  Agent-Multi (15 steps) → 77% accuracy
  Agent-UT (40 steps) → 97% accuracy

If TRUE → Non-linear relationship (UT has step-change improvement)
Implication: Self-testing is qualitatively different, not just "more work"
```

**Hypothesis C: Cost scales linearly with steps**
```
Expected:
  Agent-Corr: ~$0.05 (baseline)
  Agent-Multi: ~$0.10 (2x cost for 1.5x steps)
  Agent-UT: ~$0.30 (6x cost for 4x steps)

If TRUE → Cost predictable from trajectory length
Implication: Can estimate cost from task complexity
```

**Hypothesis D: Time and cost are tightly coupled**
```
Expected: Strong correlation (R > 0.9) between time and cost
Reasoning: Each step costs money (API calls) and takes time

If TRUE → Time is good proxy for cost
Implication: Can optimize for time to reduce cost
```

### **Implementation Code**
```python
from scipy.stats import kruskal

def plot_agent_behavior_metrics(agent_trajectories_df):
    """
    agent_trajectories_df: Columns ['scenario', 'n_steps', 'n_bash_calls',
                                     'n_edit_calls', 'duration_seconds', 'total_cost']
    """
    metrics = ['n_steps', 'tool_usage', 'duration_seconds', 'total_cost']
    titles = ['Number of Steps', 'Tool Usage (Bash + Edits)',
              'Time to Completion (s)', 'Total Cost (USD)']
    ylabels = ['Steps', 'Tool Calls', 'Seconds', 'USD']

    # Create tool usage column
    agent_trajectories_df['tool_usage'] = (agent_trajectories_df['n_bash_calls'] +
                                            agent_trajectories_df['n_edit_calls'])

    scenarios = ['Agent-Corr', 'Agent-Multi', 'Agent-UT']
    colors = ['#7FBA7A', '#4A9B4D', '#2D6B2F']

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes = axes.flatten()

    for idx, (metric, title, ylabel) in enumerate(zip(metrics, titles, ylabels)):
        ax = axes[idx]

        # Prepare data for box plot
        data_by_scenario = [
            agent_trajectories_df[agent_trajectories_df['scenario'] == s][metric].values
            for s in scenarios
        ]

        # Box plot
        bp = ax.boxplot(
            data_by_scenario,
            labels=scenarios,
            widths=0.5,
            patch_artist=True,
            showfliers=True,
            flierprops=dict(marker='o', markersize=5, alpha=0.5)
        )

        # Color boxes
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

        # Overlay individual points (jittered)
        for i, scenario in enumerate(scenarios):
            data = data_by_scenario[i]
            x = np.random.normal(i+1, 0.04, size=len(data))  # Jitter
            ax.scatter(x, data, alpha=0.3, s=30, color=colors[i], zorder=3)

        # Statistical test (Kruskal-Wallis)
        h_stat, p_value = kruskal(*data_by_scenario)
        sig_text = f'Kruskal-Wallis: H={h_stat:.2f}, p={p_value:.4f}'
        if p_value < 0.001:
            sig_text += ' ***'
        elif p_value < 0.01:
            sig_text += ' **'
        elif p_value < 0.05:
            sig_text += ' *'

        ax.text(0.5, 0.98, sig_text, transform=ax.transAxes,
                ha='center', va='top', fontsize=9,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        # Styling
        ax.set_ylabel(ylabel, fontsize=12, fontweight='bold')
        ax.set_title(title, fontsize=13, fontweight='bold', pad=10)
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    fig.suptitle('Agent Behavior Metrics: Does Agent-UT Work Harder?',
                 fontsize=16, fontweight='bold', y=0.995)

    plt.tight_layout(rect=[0, 0, 1, 0.99])
    return fig

# Expected output:
# Steps: Agent-Corr (median ~10) < Agent-Multi (median ~15) << Agent-UT (median ~40)
# Tools: Agent-Corr (~5) < Agent-Multi (~7) << Agent-UT (~20)
# Time: Agent-Corr (~60s) < Agent-Multi (~120s) << Agent-UT (~300s)
# Cost: Agent-Corr (~$0.05) < Agent-Multi (~$0.10) << Agent-UT (~$0.30)
# All p-values < 0.001 (highly significant differences)
```

---

## Graph 6: Score Distributions by UT Outcome (Split Violin Plots)

### **Purpose**
Answers **RQ2** (Multi-spec effect) and shows calibration

### **Design**
```
Split violin plots (mirrored distributions)

X-axis: Judge configuration (5 judges with continuous scores)
Y-axis: Correctness score (0.0 to 1.0)

For each judge:
  Left side (green): Tasks that PASSED unit tests
  Right side (red): Tasks that FAILED unit tests

Features:
  - Violin width shows density
  - Overlay box plots (median, quartiles)
  - Dashed line at 0.5 (decision threshold)
```

### **Expected Insights**

**Hypothesis A: Good judges have well-separated distributions**
```
Expected:
  Agent-UT: PASS distribution (mean ~0.95), FAIL distribution (mean ~0.20)
  LLM judges: PASS (mean ~0.75), FAIL (mean ~0.55) - More overlap

If TRUE → Agent-UT better calibrated (can confidently distinguish)
Implication: Agent-UT scores are more reliable for thresholding
```

**Hypothesis B: LLM judges over-estimate correctness**
```
Expected: LLM judges' FAIL distribution has median > 0.5
Reasoning: Even failed solutions get decent scores

If TRUE → LLMs too optimistic (inflate scores)
Implication: Need higher threshold (e.g., 0.7) for LLM judges
```

**Hypothesis C: Multi-spec judges better calibrated than correctness-only**
```
Expected:
  LLM-Multi distributions more separated than LLM-Corr
  Agent-Multi distributions more separated than Agent-Corr

If TRUE → Multi-spec prompting improves calibration
If FALSE → Doesn't help calibration
Implication: Multi-spec provides better confidence estimates
```

**Hypothesis D: Agent judges have tighter PASS distribution**
```
Expected: Agent-UT PASS tasks have variance < 0.05 (mostly 0.9-1.0)
LLM PASS tasks have variance > 0.10 (more spread)

If TRUE → Agents more confident on correct solutions
Implication: Agent high scores are trustworthy
```

### **Implementation Code**
```python
def plot_score_distributions_by_outcome(scores_df, ground_truth_df):
    """
    scores_df: Columns ['task_id', 'llm_corr_score', 'llm_multi_score',
                         'agent_corr_score', 'agent_multi_score', 'agent_ut_score']
    ground_truth_df: Columns ['task_id', 'passed'] (binary)
    """
    # Merge
    merged = scores_df.merge(ground_truth_df, on='task_id')

    judges = ['LLM-Corr', 'LLM-Multi', 'Agent-Corr', 'Agent-Multi', 'Agent-UT']
    score_cols = ['llm_corr_score', 'llm_multi_score', 'agent_corr_score',
                  'agent_multi_score', 'agent_ut_score']

    fig, ax = plt.subplots(figsize=(14, 8))

    positions = np.arange(len(judges)) * 2

    for idx, (judge, col) in enumerate(zip(judges, score_cols)):
        # Split data
        passed_scores = merged[merged['passed'] == 1][col].values
        failed_scores = merged[merged['passed'] == 0][col].values

        # Violin plots (split)
        parts_pass = ax.violinplot(
            [passed_scores],
            positions=[positions[idx] - 0.3],
            widths=0.5,
            showmeans=True,
            showmedians=True
        )
        parts_fail = ax.violinplot(
            [failed_scores],
            positions=[positions[idx] + 0.3],
            widths=0.5,
            showmeans=True,
            showmedians=True
        )

        # Color
        for pc in parts_pass['bodies']:
            pc.set_facecolor('#7FBA7A')
            pc.set_alpha(0.7)
        for pc in parts_fail['bodies']:
            pc.set_facecolor('#E74C3C')
            pc.set_alpha(0.7)

    # Decision threshold line
    ax.axhline(y=0.5, color='black', linestyle='--', linewidth=2,
               alpha=0.5, label='Decision Threshold (0.5)')

    # Styling
    ax.set_xticks(positions)
    ax.set_xticklabels(judges, fontsize=12)
    ax.set_ylabel('Correctness Score', fontsize=14, fontweight='bold')
    ax.set_title('Score Distributions by UT Outcome: Are Judges Well-Calibrated?',
                 fontsize=16, fontweight='bold', pad=20)
    ax.set_ylim(-0.1, 1.1)
    ax.grid(axis='y', alpha=0.3)

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#7FBA7A', alpha=0.7, label='UT PASSED'),
        Patch(facecolor='#E74C3C', alpha=0.7, label='UT FAILED'),
        plt.Line2D([0], [0], color='black', linestyle='--', label='Threshold (0.5)')
    ]
    ax.legend(handles=legend_elements, fontsize=11, loc='upper left')

    plt.tight_layout()
    return fig

# Expected output:
# Agent-UT: Strong separation (PASS centered at 0.95, FAIL at 0.20)
# LLM judges: Moderate separation (PASS at 0.75, FAIL at 0.55 - overlapping)
# Multi-spec: Slightly better separation than correctness-only
# All FAIL distributions have long tails extending to high scores (false confidence)
```

---

## Summary: Graph Selection Rationale

| Graph | Research Questions | Key Insight | Decision Impact |
|-------|-------------------|-------------|-----------------|
| **1. Accuracy Bar Chart** | RQ1, RQ3 | Agent-UT: 97% vs LLM: 73-77% | When is high accuracy worth the cost? |
| **2. Cost-Accuracy Scatter** | RQ4 | Agent-UT 24x more expensive/correct judgment | Budget allocation: LLM-Multi for volume, Agent-UT for critical tasks |
| **3. Confusion Matrices** | RQ1, RQ6 | Agents catch runtime errors (low FP), LLMs over-reject (high FN) | Where each judge excels/fails |
| **4. Multi-Aspect Correlation** | RQ5 | Style/Simplicity orthogonal (R~0.3), Robustness correlates (R~0.6) | Multi-spec provides value beyond correctness |
| **5. Agent Behavior Box Plots** | RQ7 | Agent-UT: 4x steps, 6x cost, but 20% accuracy gain | Understand what drives Agent-UT's performance |
| **6. Score Distribution Violins** | RQ2 | Agent-UT well-calibrated, LLMs over-confident on failures | Which scores to trust for thresholding |

---

## Implementation Checklist

**Data Requirements:**
- [ ] Load all 6 scenarios (use `DataLoader` class from ANALYSIS_PLAN.md)
- [ ] Merge on task_id
- [ ] Calculate binary accuracy for each judge
- [ ] Extract cost/time from agent trajectories
- [ ] Parse multi-aspect scores

**Code Files to Create:**
```
src/analysis/
├── data_loader.py           # Use template from ANALYSIS_PLAN
├── core_visualizations.py   # All 6 graph functions
└── generate_core_plots.py   # Main script to run all
```

**Expected Runtime:**
- Data loading: ~30 seconds
- All 6 graphs: ~60 seconds
- Total: < 2 minutes

**Output:**
```
outputs/analysis/figures/
├── graph1_accuracy_comparison.png
├── graph2_cost_accuracy_tradeoff.png
├── graph3_confusion_matrices.png
├── graph4_multiaspect_correlation.png
├── graph5_agent_behavior_metrics.png
└── graph6_score_distributions.png
```
