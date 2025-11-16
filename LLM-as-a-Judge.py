#!/usr/bin/env python3
"""
LLM-as-a-Judge Evaluation with OpenAI GPT-4o mini

This script:
1. Loads extracted tasks with problems, solutions, and unit test results
2. Uses OpenAI GPT-4o mini via LiteLLM to score each solution (0-1 scale)
3. Compares GPT scores with actual unit test accuracy
4. Creates visualizations showing the correlation

Requirements:
- OpenAI API key from https://platform.openai.com/api-keys
- Set as: export OPENAI_API_KEY='your-key'
- Default model: gpt-4o-mini (cheap & fast, ~60% cheaper than GPT-4o)
"""

import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple
import re
import time

# Try to import required libraries
try:
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('Agg')  # Use non-interactive backend
except ImportError:
    print("⚠️  matplotlib not installed. Installing...")
    os.system("pip install matplotlib -q")
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('Agg')

try:
    import numpy as np
except ImportError:
    print("⚠️  numpy not installed. Installing...")
    os.system("pip install numpy -q")
    import numpy as np

try:
    from litellm import completion
except ImportError:
    print("⚠️  litellm not installed. Installing...")
    os.system("pip install litellm -q")
    from litellm import completion


class LLMJudgeEvaluator:
    """Evaluates code solutions using LLM judge with 0-1 scoring."""
    
    def __init__(self, extracted_tasks_json: str, api_key: str = None, model: str = "gpt-4o-mini"):
        """
        Initialize evaluator.
        
        Args:
            extracted_tasks_json: Path to extracted_tasks.json
            api_key: OpenAI API key (or set OPENAI_API_KEY env var)
            model: OpenAI model to use (default: gpt-4o-mini)
        """
        # Load data
        with open(extracted_tasks_json) as f:
            self.data = json.load(f)
        
        self.tasks = self.data['tasks']
        self.model = model
        
        # Set OpenAI API key
        self.api_key = api_key or os.getenv('OPENAI_API_KEY')
        if self.api_key:
            os.environ['OPENAI_API_KEY'] = self.api_key
        
        self.results = []
        
        print(f"✓ Using OpenAI model: {model}")
        print(f"✓ Loaded {len(self.tasks)} tasks")
        print(f"  Summary: {self.data['summary']}")
    
    def create_judge_prompt(self, task: Dict) -> str:
        """
        Create prompt asking for 0-1 score.
        
        Score interpretation:
        - 1.0: Perfect solution, handles all cases
        - 0.8-0.9: Very good, minor potential issues
        - 0.6-0.7: Good, but has some issues
        - 0.4-0.5: Partial solution, significant issues
        - 0.2-0.3: Poor solution, major flaws
        - 0.0-0.1: Completely wrong or doesn't address problem
        """
        # Truncate very long problem descriptions
        problem = task['problem_description']
        if len(problem) > 2000:
            problem = problem[:2000] + "\n... (truncated for brevity)"
        
        # Truncate very long solutions
        solution = task['code_solution']
        if len(solution) > 3000:
            solution = solution[:3000] + "\n... (truncated)"
        
        prompt = f"""You are an expert code reviewer. Evaluate the correctness and quality of the following code solution.

**Problem:**
{problem}

**Code Solution:**
```python
{solution}
```

**Your Task:**
Evaluate this solution and provide a confidence score from 0.0 to 1.0 indicating how likely it is to correctly solve the problem.

**Scoring Guidelines:**
- 1.0: Perfect solution, correctly handles all cases including edge cases
- 0.8-0.9: Very strong solution, likely correct with minor potential concerns
- 0.6-0.7: Good solution, but has some notable issues or missing edge cases
- 0.4-0.5: Partial solution with significant correctness concerns
- 0.2-0.3: Poor solution with major logical flaws
- 0.0-0.1: Completely incorrect or doesn't address the problem

**Analysis Criteria:**
1. Does the algorithm logic correctly solve the problem?
2. Are edge cases handled (empty inputs, boundaries, special values)?
3. Is the time/space complexity acceptable for given constraints?
4. Are there any obvious bugs or logical errors?

**Response Format:**
Score: [0.0-1.0]
Reasoning: [2-3 sentences explaining your score]

Provide ONLY the score and reasoning, nothing else."""

        return prompt
    
    def parse_score(self, response_text: str) -> float:
        """
        Parse score from LLM response.
        
        Args:
            response_text: Raw LLM response
            
        Returns:
            Score between 0.0 and 1.0, or -1.0 if parsing failed
        """
        # Look for "Score: X.X" pattern
        score_match = re.search(r'Score:\s*([0-9]*\.?[0-9]+)', response_text, re.IGNORECASE)
        
        if score_match:
            try:
                score = float(score_match.group(1))
                # Clamp to [0, 1]
                score = max(0.0, min(1.0, score))
                return score
            except ValueError:
                pass
        
        # Try to find any number between 0 and 1 in the first line
        first_line = response_text.split('\n')[0]
        number_match = re.search(r'\b([0-1]\.?\d*)\b', first_line)
        if number_match:
            try:
                score = float(number_match.group(1))
                if 0.0 <= score <= 1.0:
                    return score
            except ValueError:
                pass
        
        print(f"⚠️  Failed to parse score from: {response_text[:100]}")
        return -1.0
    
    def evaluate_with_llm(self, task: Dict, model: str = None) -> Tuple[float, str]:
        """
        Evaluate a task using LLM judge via LiteLLM.
        
        Args:
            task: Task dictionary
            model: Model identifier for LiteLLM (uses self.model if not specified)
            
        Returns:
            Tuple of (score, reasoning_text)
        """
        if model is None:
            model = self.model
            
        try:
            prompt = self.create_judge_prompt(task)
            
            # Call LLM via LiteLLM
            response = completion(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=500,
                temperature=0.3,
                api_key=self.api_key
            )
            
            response_text = response.choices[0].message.content
            score = self.parse_score(response_text)
            
            return score, response_text
            
        except Exception as e:
            print(f"✗ Error calling LLM: {e}")
            return -1.0, f"Error: {str(e)}"
    
    def compute_unit_test_accuracy(self, task: Dict) -> float:
        """
        Compute actual unit test accuracy (passed_tests / total_tests).
        
        Args:
            task: Task dictionary
            
        Returns:
            Accuracy between 0.0 and 1.0
        """
        if task['total_tests'] == 0:
            return 0.0
        
        return task['passed_tests'] / task['total_tests']
    
    def run_evaluation(self, sample_size: int = None) -> List[Dict]:
        """
        Run evaluation on tasks.
        
        Args:
            sample_size: Number of tasks to evaluate (None = all tasks)
            
        Returns:
            List of evaluation results
        """
        tasks_to_eval = self.tasks[:sample_size] if sample_size else self.tasks
        
        print(f"\n{'='*80}")
        print(f"Running LLM-as-a-Judge evaluation on {len(tasks_to_eval)} tasks...")
        print(f"Model: {self.model}")
        print(f"{'='*80}\n")
        
        for i, task in enumerate(tasks_to_eval):
            task_id = task['task_id']
            
            # Get LLM score
            llm_score, reasoning = self.evaluate_with_llm(task)
            
            # Get actual accuracy
            actual_accuracy = self.compute_unit_test_accuracy(task)
            
            # Record result
            result = {
                'task_id': task_id,
                'llm_score': llm_score,
                'actual_accuracy': actual_accuracy,
                'passed_tests': task['passed_tests'],
                'total_tests': task['total_tests'],
                'unit_test_status': task['unit_test_status'],
                'reasoning': reasoning,
                'model': self.model
            }
            self.results.append(result)
            
            # Print progress
            diff = abs(llm_score - actual_accuracy)
            match_symbol = "✓" if diff < 0.3 else "⚠" if diff < 0.5 else "✗"
            
            print(f"[{i+1}/{len(tasks_to_eval)}] {match_symbol} {task_id}")
            print(f"    LLM Score: {llm_score:.2f} | Actual: {actual_accuracy:.2f} | Diff: {diff:.2f}")
            
            # Rate limiting
            time.sleep(0.5)  # Small delay between requests
        
        return self.results
    
    def compute_metrics(self) -> Dict:
        """Compute evaluation metrics."""
        if not self.results:
            return {}
        
        # Filter out failed parses
        valid_results = [r for r in self.results if r['llm_score'] >= 0]
        
        if not valid_results:
            return {'error': 'No valid results'}
        
        llm_scores = np.array([r['llm_score'] for r in valid_results])
        actual_scores = np.array([r['actual_accuracy'] for r in valid_results])
        
        # Compute correlation
        correlation = np.corrcoef(llm_scores, actual_scores)[0, 1]
        
        # Compute MAE (Mean Absolute Error)
        mae = np.mean(np.abs(llm_scores - actual_scores))
        
        # Compute RMSE (Root Mean Squared Error)
        rmse = np.sqrt(np.mean((llm_scores - actual_scores) ** 2))
        
        # Binary classification metrics (using 0.5 threshold)
        llm_binary = (llm_scores >= 0.5).astype(int)
        actual_binary = (actual_scores >= 0.5).astype(int)
        
        accuracy = np.mean(llm_binary == actual_binary)
        
        return {
            'correlation': correlation,
            'mae': mae,
            'rmse': rmse,
            'binary_accuracy': accuracy,
            'n_valid': len(valid_results),
            'n_total': len(self.results)
        }
    
    def create_visualizations(self, output_dir: str = "."):
        """
        Create visualizations comparing LLM scores with actual accuracy.
        
        Args:
            output_dir: Directory to save plots
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        
        # Filter valid results
        valid_results = [r for r in self.results if r['llm_score'] >= 0]
        
        if not valid_results:
            print("No valid results to visualize")
            return
        
        llm_scores = [r['llm_score'] for r in valid_results]
        actual_scores = [r['actual_accuracy'] for r in valid_results]
        
        # Create figure with multiple subplots
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        fig.suptitle('LLM-as-a-Judge vs. Unit Test Accuracy', fontsize=16, fontweight='bold')
        
        # Plot 1: Scatter plot with perfect correlation line
        ax1 = axes[0, 0]
        ax1.scatter(llm_scores, actual_scores, alpha=0.6, s=100, edgecolors='black', linewidth=0.5)
        ax1.plot([0, 1], [0, 1], 'r--', linewidth=2, label='Perfect Correlation')
        ax1.set_xlabel('LLM Judge Score', fontsize=12)
        ax1.set_ylabel('Actual Unit Test Accuracy', fontsize=12)
        ax1.set_title('Score Comparison', fontsize=14, fontweight='bold')
        ax1.set_xlim(-0.05, 1.05)
        ax1.set_ylim(-0.05, 1.05)
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # Add correlation coefficient
        metrics = self.compute_metrics()
        ax1.text(0.05, 0.95, f"Correlation: {metrics['correlation']:.3f}\nMAE: {metrics['mae']:.3f}", 
                transform=ax1.transAxes, fontsize=11, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        # Plot 2: Histogram of differences
        ax2 = axes[0, 1]
        differences = np.array(llm_scores) - np.array(actual_scores)
        ax2.hist(differences, bins=20, edgecolor='black', alpha=0.7)
        ax2.axvline(x=0, color='r', linestyle='--', linewidth=2, label='Zero Difference')
        ax2.set_xlabel('Difference (LLM Score - Actual)', fontsize=12)
        ax2.set_ylabel('Frequency', fontsize=12)
        ax2.set_title('Distribution of Prediction Errors', fontsize=14, fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3, axis='y')
        
        # Plot 3: Box plot by unit test status
        ax3 = axes[1, 0]
        passed_scores = [r['llm_score'] for r in valid_results if r['unit_test_status'] == 'PASSED']
        failed_scores = [r['llm_score'] for r in valid_results if r['unit_test_status'] == 'FAILED']
        
        box_data = [passed_scores, failed_scores]
        box_labels = [f'PASSED\n(n={len(passed_scores)})', f'FAILED\n(n={len(failed_scores)})']
        
        bp = ax3.boxplot(box_data, labels=box_labels, patch_artist=True)
        bp['boxes'][0].set_facecolor('lightgreen')
        bp['boxes'][1].set_facecolor('lightcoral')
        
        ax3.set_ylabel('LLM Judge Score', fontsize=12)
        ax3.set_title('LLM Scores by Test Status', fontsize=14, fontweight='bold')
        ax3.grid(True, alpha=0.3, axis='y')
        ax3.set_ylim(-0.05, 1.05)
        
        # Plot 4: Accuracy ranges
        ax4 = axes[1, 1]
        
        # Group by actual accuracy ranges
        ranges = [(0.0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.0)]
        range_labels = ['0.0-0.2', '0.2-0.4', '0.4-0.6', '0.6-0.8', '0.8-1.0']
        
        range_llm_means = []
        range_counts = []
        
        for low, high in ranges:
            in_range = [r for r in valid_results if low <= r['actual_accuracy'] < high or 
                       (high == 1.0 and r['actual_accuracy'] == 1.0)]
            if in_range:
                range_llm_means.append(np.mean([r['llm_score'] for r in in_range]))
                range_counts.append(len(in_range))
            else:
                range_llm_means.append(0)
                range_counts.append(0)
        
        x_pos = np.arange(len(range_labels))
        bars = ax4.bar(x_pos, range_llm_means, alpha=0.7, edgecolor='black')
        
        # Color bars
        colors = plt.cm.RdYlGn(np.linspace(0.2, 0.8, len(bars)))
        for bar, color in zip(bars, colors):
            bar.set_color(color)
        
        ax4.set_xlabel('Actual Accuracy Range', fontsize=12)
        ax4.set_ylabel('Mean LLM Score', fontsize=12)
        ax4.set_title('LLM Scores by Actual Accuracy Range', fontsize=14, fontweight='bold')
        ax4.set_xticks(x_pos)
        ax4.set_xticklabels(range_labels, rotation=45, ha='right')
        ax4.set_ylim(0, 1.05)
        ax4.grid(True, alpha=0.3, axis='y')
        
        # Add count labels on bars
        for i, (bar, count) in enumerate(zip(bars, range_counts)):
            height = bar.get_height()
            if count > 0:
                ax4.text(bar.get_x() + bar.get_width()/2., height,
                        f'n={count}', ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        
        # Save figure
        output_path = output_dir / 'llm_judge_comparison.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"\n✓ Visualization saved to: {output_path}")
        
        plt.close()
    
    def print_report(self):
        """Print evaluation report."""
        metrics = self.compute_metrics()
        
        print(f"\n{'='*80}")
        print("EVALUATION REPORT")
        print(f"{'='*80}")
        
        print(f"\nDataset:")
        print(f"  Total tasks evaluated: {metrics['n_total']}")
        print(f"  Valid evaluations: {metrics['n_valid']}")
        
        print(f"\nCorrelation Metrics:")
        print(f"  Pearson Correlation: {metrics['correlation']:.3f}")
        print(f"  Mean Absolute Error: {metrics['mae']:.3f}")
        print(f"  Root Mean Squared Error: {metrics['rmse']:.3f}")
        
        print(f"\nBinary Classification (threshold=0.5):")
        print(f"  Accuracy: {metrics['binary_accuracy']:.2%}")
        
        # Show some examples
        print(f"\n{'='*80}")
        print("SAMPLE RESULTS (first 5)")
        print(f"{'='*80}")
        
        for i, result in enumerate(self.results[:5]):
            print(f"\n{i+1}. {result['task_id']}")
            print(f"   LLM Score: {result['llm_score']:.2f}")
            print(f"   Actual: {result['actual_accuracy']:.2f} ({result['passed_tests']}/{result['total_tests']} tests)")
            print(f"   Status: {result['unit_test_status']}")
            print(f"   Reasoning: {result['reasoning'][:150]}...")
        
        print(f"\n{'='*80}\n")
    
    def save_results(self, output_file: str = "llm_judge_results.json"):
        """Save results to JSON."""
        metrics = self.compute_metrics()
        
        output = {
            'metrics': metrics,
            'summary': self.data['summary'],
            'results': self.results
        }
        
        with open(output_file, 'w') as f:
            json.dump(output, f, indent=2)
        
        print(f"✓ Results saved to: {output_file}")


def main():
    """Main entry point."""
    
    tasks_json = "./extracted_tasks.json"
    sample_size = None
    model = "gpt-4o-mini"
    
    print("="*80)
    print("LLM-AS-A-JUDGE EVALUATION WITH OPENAI")
    print("="*80)
    print(f"Model: {model}")
    print()
    
    # Run evaluation
    evaluator = LLMJudgeEvaluator(tasks_json, model=model)
    evaluator.run_evaluation(sample_size=sample_size)
    
    # Generate report and visualizations
    evaluator.print_report()
    evaluator.create_visualizations()
    evaluator.save_results("llm_judge_results.json")
    
    print("\n" + "="*80)
    print("✅ EVALUATION COMPLETE")
    print("="*80)
    print("\nGenerated files:")
    print("  - llm_judge_results.json (detailed results)")
    print("  - llm_judge_comparison.png (visualizations)")
    print("="*80)


if __name__ == "__main__":
    main()