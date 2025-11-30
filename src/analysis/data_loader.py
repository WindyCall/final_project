"""DataLoader class for loading and merging all 6 judge scenario results."""

import json
import pandas as pd
import re
from pathlib import Path
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')


class DataLoader:
    """Load and merge data from all 6 judge scenarios."""

    def __init__(self, base_path="outputs/results", agent_base="agent_judge_tasks_results"):
        """
        Initialize DataLoader with paths to data directories.

        Args:
            base_path: Path to LLM judge results and ground truth
            agent_base: Path to agent judge results directories
        """
        self.base_path = Path(base_path)
        self.agent_base = Path(agent_base)

    def load_unit_tests(self):
        """
        Load ground truth unit test results (Scenario 1).

        Returns:
            DataFrame with task_id and unit test outcomes
        """
        path = self.base_path / "unit_test_summary.json"
        with open(path) as f:
            data = json.load(f)

        df = pd.DataFrame.from_dict(data, orient='index')
        df['task_id'] = df.index
        df['passed'] = (df['status'] == 'PASSED').astype(int)
        df['pass_rate'] = df['passed_tests'] / df['total_tests']

        return df

    def load_llm_correctness(self):
        """
        Load LLM correctness-only judge results (Scenario 2).

        Returns:
            DataFrame with task_id and correctness scores
        """
        path = self.base_path / "correctness" / "correctness_llm_judge_results.json"
        with open(path) as f:
            data = json.load(f)

        results = data.get('results', [])
        df = pd.DataFrame(results)

        # Binary prediction based on threshold 0.5
        if 'correctness_score' in df.columns:
            df['llm_corr_pred'] = (df['correctness_score'] > 0.5).astype(int)

        return df

    def load_llm_multi(self):
        """
        Load LLM multi-aspect judge results (Scenario 3).

        Returns:
            DataFrame with task_id and 4-dimensional scores
        """
        path = self.base_path / "multi_aspect" / "multi_aspect_llm_judge_results.json"
        with open(path) as f:
            data = json.load(f)

        results = data.get('results', [])
        df = pd.DataFrame(results)

        # Binary prediction based on correctness score
        if 'correctness_score' in df.columns:
            df['llm_multi_pred'] = (df['correctness_score'] > 0.5).astype(int)

        return df

    def parse_agent_scores_from_evaluation_file(self, file_text):
        """
        Extract score from evaluation.txt file content.

        Expected format:
        'Score: 1.0\\nReasoning: ...'

        Args:
            file_text: String containing evaluation file content

        Returns:
            Dictionary with extracted score
        """
        scores = {}

        # Try to find the main score from evaluation.txt
        score_match = re.search(r'Score:\s*([\d.]+)', file_text, re.IGNORECASE)
        if score_match:
            scores['correctness'] = float(score_match.group(1))

        # Also try to find multi-aspect scores
        patterns = {
            'correctness': r'Correctness[_\s]*Score:\s*([\d.]+)',
            'style': r'Style[_\s]*Score:\s*([\d.]+)',
            'simplicity': r'Simplicity[_\s]*Score:\s*([\d.]+)',
            'robustness': r'Robustness[_\s]*Score:\s*([\d.]+)'
        }

        for aspect, pattern in patterns.items():
            match = re.search(pattern, file_text, re.IGNORECASE)
            if match and aspect != 'correctness':  # Don't overwrite correctness if already found
                scores[aspect] = float(match.group(1))

        return scores

    def parse_agent_trajectory(self, traj, task_id):
        """
        Parse a single agent trajectory JSON to extract scores and behavior metrics.

        Args:
            traj: Trajectory JSON dictionary
            task_id: Task identifier

        Returns:
            Dictionary with parsed trajectory data
        """
        # Try main structure first
        steps = traj.get('steps', [])

        if len(steps) == 0:
            return {
                'task_id': task_id,
                'n_steps': 0,
                'n_bash_calls': 0,
                'n_edit_calls': 0,
                'total_cost': 0.0,
                'duration_seconds': 0.0
            }

        # Count steps and analyze tool usage
        n_steps = len(steps)
        n_bash = 0
        n_edits = 0
        total_cost = 0.0

        # Search for evaluation scores
        scores = {}
        evaluation_found = False

        # Parse timestamps for duration
        started_at = None
        finished_at = None

        for step in steps:
            # Extract timestamp
            if 'timestamp' in step:
                ts = step['timestamp']
                if started_at is None:
                    started_at = ts
                finished_at = ts

            # Count tool usage from tool_calls
            if 'tool_calls' in step:
                for tool_call in step['tool_calls']:
                    func_name = tool_call.get('function_name', '').lower()
                    if 'bash' in func_name or 'execute' in func_name or 'run' in func_name:
                        n_bash += 1
                    if 'editor' in func_name or 'write' in func_name or 'edit' in func_name:
                        n_edits += 1

                    # Look for evaluation.txt creation
                    if not evaluation_found:
                        args = tool_call.get('arguments', {})
                        if isinstance(args, dict):
                            path = args.get('path', '')
                            if 'evaluation.txt' in path:
                                file_text = args.get('file_text', '')
                                parsed_scores = self.parse_agent_scores_from_evaluation_file(file_text)
                                if parsed_scores:
                                    scores.update(parsed_scores)
                                    evaluation_found = True

            # Extract cost from metrics
            if 'metrics' in step:
                metrics = step['metrics']
                if isinstance(metrics, dict):
                    if 'cost_usd' in metrics:
                        total_cost += metrics['cost_usd']

        # Calculate duration
        duration = 0.0
        if started_at and finished_at:
            try:
                start = datetime.fromisoformat(started_at.replace('Z', '+00:00'))
                end = datetime.fromisoformat(finished_at.replace('Z', '+00:00'))
                duration = (end - start).total_seconds()
            except:
                pass

        result = {
            'task_id': task_id,
            'n_steps': n_steps,
            'n_bash_calls': n_bash,
            'n_edit_calls': n_edits,
            'total_cost': total_cost,
            'duration_seconds': duration,
            **scores  # Unpack scores dict
        }

        return result

    def load_agent_results(self, judge_type):
        """
        Load agent results from trajectory files.

        Args:
            judge_type: One of 'agent_correctness_result', 'agent_multi_spec_result',
                       or 'agent_unit_test_result'

        Returns:
            DataFrame with parsed trajectory data
        """
        trajectories = []
        agent_path = self.agent_base / judge_type

        print(f"Loading agent results from: {agent_path}")

        for task_dir in agent_path.glob("*/"):
            if task_dir.is_dir() and task_dir.name not in ['__pycache__', '.git']:
                # Look for trajectory.json
                traj_file = task_dir / "agent" / "trajectory.json"

                if not traj_file.exists():
                    # Try alternative location
                    traj_file = task_dir / "trajectory.json"

                if traj_file.exists():
                    try:
                        with open(traj_file, encoding='utf-8') as f:
                            traj = json.load(f)

                        # Parse trajectory - use directory name as task_id initially
                        parsed = self.parse_agent_trajectory(traj, task_dir.name)
                        trajectories.append(parsed)
                    except Exception as e:
                        print(f"Error parsing {traj_file}: {e}")

        df = pd.DataFrame(trajectories)
        print(f"Loaded {len(df)} trajectories for {judge_type}")

        # Normalize task_id: remove the session ID suffix (last __<id> part)
        # e.g., "1873_d__ewd5mxi__z3M3aK9" -> "1873_d__ewd5mxi"
        # Also convert to lowercase for case-insensitive matching
        if len(df) > 0:
            df['task_id'] = df['task_id'].apply(lambda x: '__'.join(x.split('__')[:-1]).lower())

        # Add binary prediction based on correctness score
        if 'correctness' in df.columns:
            df['agent_pred'] = (df['correctness'] > 0.5).astype(int)

        return df

    def normalize_task_id(self, task_id):
        """
        Normalize task_id format to handle different naming conventions.

        Args:
            task_id: Original task identifier

        Returns:
            Normalized task_id string
        """
        # Keep original format but ensure consistency
        return str(task_id).strip()

    def merge_all(self):
        """
        Merge all 6 data sources on task_id.

        Returns:
            DataFrame with all scenarios merged
        """
        print("Loading all data sources...")

        # Load ground truth
        print("\n1. Loading ground truth (unit tests)...")
        ut = self.load_unit_tests()
        print(f"   Loaded {len(ut)} tasks")

        # Load LLM judges
        print("\n2. Loading LLM-Correctness judge...")
        llm_corr = self.load_llm_correctness()
        print(f"   Loaded {len(llm_corr)} evaluations")

        print("\n3. Loading LLM-Multi judge...")
        llm_multi = self.load_llm_multi()
        print(f"   Loaded {len(llm_multi)} evaluations")

        # Load Agent judges
        print("\n4. Loading Agent-Correctness judge...")
        agent_corr = self.load_agent_results('agent_correctness_result')

        print("\n5. Loading Agent-Multi judge...")
        agent_multi = self.load_agent_results('agent_multi_spec_result')

        print("\n6. Loading Agent-UnitTest judge...")
        agent_ut = self.load_agent_results('agent_unit_test_result')

        # Normalize all task_ids to lowercase for matching
        for df in [ut, llm_corr, llm_multi]:
            if 'task_id' in df.columns:
                df['task_id'] = df['task_id'].str.lower()

        # Start with unit tests as base
        merged = ut[['task_id', 'passed', 'pass_rate', 'passed_tests', 'total_tests']].copy()

        # Merge LLM-Correctness
        if len(llm_corr) > 0:
            llm_corr_cols = ['task_id', 'correctness_score', 'llm_corr_pred']
            llm_corr_cols = [c for c in llm_corr_cols if c in llm_corr.columns]
            merged = merged.merge(
                llm_corr[llm_corr_cols],
                on='task_id',
                how='left',
                suffixes=('', '_llm_corr')
            )
            merged.rename(columns={'correctness_score': 'llm_corr_score'}, inplace=True)

        # Merge LLM-Multi
        if len(llm_multi) > 0:
            llm_multi_cols = ['task_id', 'correctness_score', 'style_score',
                             'simplicity_score', 'robustness_score', 'llm_multi_pred']
            llm_multi_cols = [c for c in llm_multi_cols if c in llm_multi.columns]
            merged = merged.merge(
                llm_multi[llm_multi_cols],
                on='task_id',
                how='left',
                suffixes=('', '_llm_multi')
            )
            merged.rename(columns={
                'correctness_score': 'llm_multi_correctness',
                'style_score': 'llm_multi_style',
                'simplicity_score': 'llm_multi_simplicity',
                'robustness_score': 'llm_multi_robustness'
            }, inplace=True)

        # Merge Agent-Correctness
        if len(agent_corr) > 0:
            agent_corr_cols = ['task_id', 'correctness', 'n_steps', 'n_bash_calls',
                              'n_edit_calls', 'total_cost', 'duration_seconds', 'agent_pred']
            agent_corr_cols = [c for c in agent_corr_cols if c in agent_corr.columns]
            merged = merged.merge(
                agent_corr[agent_corr_cols],
                on='task_id',
                how='left',
                suffixes=('', '_agent_corr')
            )
            merged.rename(columns={
                'correctness': 'agent_corr_score',
                'n_steps': 'agent_corr_steps',
                'n_bash_calls': 'agent_corr_bash',
                'n_edit_calls': 'agent_corr_edits',
                'total_cost': 'agent_corr_cost',
                'duration_seconds': 'agent_corr_time',
                'agent_pred': 'agent_corr_pred'
            }, inplace=True)

        # Merge Agent-Multi
        if len(agent_multi) > 0:
            agent_multi_cols = ['task_id', 'correctness', 'style', 'simplicity', 'robustness',
                               'n_steps', 'n_bash_calls', 'n_edit_calls',
                               'total_cost', 'duration_seconds', 'agent_pred']
            agent_multi_cols = [c for c in agent_multi_cols if c in agent_multi.columns]
            merged = merged.merge(
                agent_multi[agent_multi_cols],
                on='task_id',
                how='left',
                suffixes=('', '_agent_multi')
            )
            merged.rename(columns={
                'correctness': 'agent_multi_correctness',
                'style': 'agent_multi_style',
                'simplicity': 'agent_multi_simplicity',
                'robustness': 'agent_multi_robustness',
                'n_steps': 'agent_multi_steps',
                'n_bash_calls': 'agent_multi_bash',
                'n_edit_calls': 'agent_multi_edits',
                'total_cost': 'agent_multi_cost',
                'duration_seconds': 'agent_multi_time',
                'agent_pred': 'agent_multi_pred'
            }, inplace=True)

        # Merge Agent-UT
        if len(agent_ut) > 0:
            agent_ut_cols = ['task_id', 'correctness', 'style', 'simplicity', 'robustness',
                            'n_steps', 'n_bash_calls', 'n_edit_calls',
                            'total_cost', 'duration_seconds', 'agent_pred']
            agent_ut_cols = [c for c in agent_ut_cols if c in agent_ut.columns]
            merged = merged.merge(
                agent_ut[agent_ut_cols],
                on='task_id',
                how='left',
                suffixes=('', '_agent_ut')
            )
            merged.rename(columns={
                'correctness': 'agent_ut_correctness',
                'style': 'agent_ut_style',
                'simplicity': 'agent_ut_simplicity',
                'robustness': 'agent_ut_robustness',
                'n_steps': 'agent_ut_steps',
                'n_bash_calls': 'agent_ut_bash',
                'n_edit_calls': 'agent_ut_edits',
                'total_cost': 'agent_ut_cost',
                'duration_seconds': 'agent_ut_time',
                'agent_pred': 'agent_ut_pred'
            }, inplace=True)

        print(f"\n✓ Merged dataset: {len(merged)} tasks × {len(merged.columns)} columns")
        print(f"  Columns: {list(merged.columns)}")

        return merged


if __name__ == "__main__":
    # Test data loading
    loader = DataLoader()

    print("="*80)
    print("TESTING DATA LOADER")
    print("="*80)

    # Test individual loaders
    print("\nTesting individual loaders...")
    ut = loader.load_unit_tests()
    print(f"✓ Unit tests: {len(ut)} tasks")

    llm_corr = loader.load_llm_correctness()
    print(f"✓ LLM-Correctness: {len(llm_corr)} evaluations")

    llm_multi = loader.load_llm_multi()
    print(f"✓ LLM-Multi: {len(llm_multi)} evaluations")

    # Test merge
    print("\nTesting full merge...")
    merged = loader.merge_all()

    print(f"\n✓ Final merged dataset: {merged.shape}")
    print(f"\nFirst 5 rows:")
    print(merged.head())

    print(f"\nData types:")
    print(merged.dtypes)

    print("\n" + "="*80)
    print("DATA LOADER TEST COMPLETE")
    print("="*80)
