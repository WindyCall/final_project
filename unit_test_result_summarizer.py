#!/usr/bin/env python3
"""
Summarize unit test results from LiveCodeBench task outputs.

This script traverses the FINAL_PROJECT directory structure, finds all
test-stdout.txt files, and extracts pass/fail information for each task.
"""

import os
import re
from pathlib import Path
from typing import Dict, List, Tuple
import json


class UnitTestSummary:
    def __init__(self, project_root: str = "./2025-11-04__15-32-48"):
        """
        Initialize the summarizer.
        
        Args:
            project_root: Root directory containing task folders
        """
        self.project_root = Path(project_root)
        self.results = []
        
    def extract_final_result(self, content: str) -> Dict:
        """
        Extract the final test result from test-stdout.txt content.
        
        Args:
            content: Full content of test-stdout.txt
            
        Returns:
            Dictionary with test results including:
            - status: 'PASSED' or 'FAILED'
            - total_tests: total number of tests
            - passed_tests: number of passed tests
            - failed_tests: number of failed tests
            - final_line: the final result line
        """
        lines = content.strip().split('\n')
        
        # Look for the final result line (last non-empty line)
        final_line = None
        for line in reversed(lines):
            if line.strip():
                final_line = line.strip()
                break
        
        if not final_line:
            return {
                'status': 'UNKNOWN',
                'total_tests': 0,
                'passed_tests': 0,
                'failed_tests': 0,
                'final_line': '',
                'error': 'No result line found'
            }
        
        # Parse PASSED pattern: "🎉 TASK PASSED: All X tests passed!"
        passed_pattern = r'(?:🎉|✓)?\s*TASK PASSED:\s*All\s+(\d+)\s+tests?\s+passed!'
        passed_match = re.search(passed_pattern, final_line, re.IGNORECASE)
        
        if passed_match:
            total = int(passed_match.group(1))
            return {
                'status': 'PASSED',
                'total_tests': total,
                'passed_tests': total,
                'failed_tests': 0,
                'final_line': final_line
            }
        
        # Parse FAILED pattern: "❌ TASK FAILED: X out of Y tests failed!"
        failed_pattern = r'(?:❌|✗)?\s*TASK FAILED:\s*(\d+)\s+out of\s+(\d+)\s+tests?\s+failed!'
        failed_match = re.search(failed_pattern, final_line, re.IGNORECASE)
        
        if failed_match:
            failed = int(failed_match.group(1))
            total = int(failed_match.group(2))
            return {
                'status': 'FAILED',
                'total_tests': total,
                'passed_tests': total - failed,
                'failed_tests': failed,
                'final_line': final_line
            }
        
        # Alternative pattern: check for pytest summary
        # "=========================== 14 passed in 0.01s ==========================="
        pytest_passed = r'=+\s*(\d+)\s+passed'
        pytest_match = re.search(pytest_passed, final_line)
        
        if pytest_match:
            total = int(pytest_match.group(1))
            return {
                'status': 'PASSED',
                'total_tests': total,
                'passed_tests': total,
                'failed_tests': 0,
                'final_line': final_line
            }
        
        # Check if there's a pytest failed line
        pytest_failed = r'=+\s*(\d+)\s+failed.*?(\d+)\s+passed'
        pytest_fail_match = re.search(pytest_failed, final_line)
        
        if pytest_fail_match:
            failed = int(pytest_fail_match.group(1))
            passed = int(pytest_fail_match.group(2))
            return {
                'status': 'FAILED',
                'total_tests': failed + passed,
                'passed_tests': passed,
                'failed_tests': failed,
                'final_line': final_line
            }
        
        return {
            'status': 'UNKNOWN',
            'total_tests': 0,
            'passed_tests': 0,
            'failed_tests': 0,
            'final_line': final_line,
            'error': 'Could not parse result line'
        }
    
    def process_task(self, task_folder: Path) -> Dict:
        """
        Process a single task folder and extract test results.
        
        Args:
            task_folder: Path to task folder (e.g., 1873_d__ewd5mxi)
            
        Returns:
            Dictionary with task information and test results
        """
        task_id = task_folder.name
        test_file = task_folder / "verifier" / "test-stdout.txt"
        
        if not test_file.exists():
            return {
                'task_id': task_id,
                'status': 'FILE_NOT_FOUND',
                'error': f'test-stdout.txt not found at {test_file}'
            }
        
        try:
            content = test_file.read_text(encoding='utf-8', errors='ignore')
            result = self.extract_final_result(content)
            result['task_id'] = task_id
            result['test_file_path'] = str(test_file)
            return result
            
        except Exception as e:
            return {
                'task_id': task_id,
                'status': 'ERROR',
                'error': str(e)
            }
    
    def scan_all_tasks(self):
        """
        Scan all task folders in the project root and extract results.
        """
        # Look for folders that match task pattern (contain underscore)
        task_folders = []
        
        for item in self.project_root.iterdir():
            if item.is_dir() and '_' in item.name:
                # Check if it has a verifier subfolder
                verifier_path = item / "verifier"
                if verifier_path.exists() and verifier_path.is_dir():
                    task_folders.append(item)
        
        print(f"Found {len(task_folders)} task folders")
        
        for task_folder in sorted(task_folders):
            result = self.process_task(task_folder)
            self.results.append(result)
    
    def generate_summary(self) -> Dict:
        """
        Generate aggregate summary statistics.
        
        Returns:
            Dictionary with summary statistics
        """
        total_tasks = len(self.results)
        passed_tasks = sum(1 for r in self.results if r['status'] == 'PASSED')
        failed_tasks = sum(1 for r in self.results if r['status'] == 'FAILED')
        error_tasks = sum(1 for r in self.results if r['status'] not in ['PASSED', 'FAILED'])
        
        total_tests = sum(r.get('total_tests', 0) for r in self.results)
        total_passed = sum(r.get('passed_tests', 0) for r in self.results)
        total_failed = sum(r.get('failed_tests', 0) for r in self.results)
        
        return {
            'total_tasks': total_tasks,
            'passed_tasks': passed_tasks,
            'failed_tasks': failed_tasks,
            'error_tasks': error_tasks,
            'task_pass_rate': f"{passed_tasks}/{total_tasks} ({100*passed_tasks/total_tasks:.1f}%)" if total_tasks > 0 else "N/A",
            'total_test_cases': total_tests,
            'total_passed_tests': total_passed,
            'total_failed_tests': total_failed,
            'test_case_pass_rate': f"{total_passed}/{total_tests} ({100*total_passed/total_tests:.1f}%)" if total_tests > 0 else "N/A"
        }
    
    def print_results(self):
        """
        Print formatted results to console.
        """
        print("\n" + "="*80)
        print("UNIT TEST RESULTS SUMMARY")
        print("="*80 + "\n")
        
        # Print per-task results
        print("Per-Task Results:")
        print("-" * 80)
        print(f"{'Task ID':<30} {'Status':<10} {'Tests':<15} {'Result'}")
        print("-" * 80)
        
        for result in self.results:
            task_id = result['task_id']
            status = result['status']
            
            if status == 'PASSED':
                tests_str = f"{result['passed_tests']}/{result['total_tests']}"
                result_str = "✓ All passed"
                status_display = "✓ PASSED"
            elif status == 'FAILED':
                tests_str = f"{result['passed_tests']}/{result['total_tests']}"
                result_str = f"✗ {result['failed_tests']} failed"
                status_display = "✗ FAILED"
            else:
                tests_str = "N/A"
                result_str = result.get('error', 'Unknown error')[:30]
                status_display = "⚠ ERROR"
            
            print(f"{task_id:<30} {status_display:<10} {tests_str:<15} {result_str}")
        
        # Print summary statistics
        summary = self.generate_summary()
        print("\n" + "="*80)
        print("AGGREGATE STATISTICS")
        print("="*80)
        print(f"Total Tasks:              {summary['total_tasks']}")
        print(f"Passed Tasks:             {summary['passed_tasks']}")
        print(f"Failed Tasks:             {summary['failed_tasks']}")
        print(f"Error Tasks:              {summary['error_tasks']}")
        print(f"Task Pass Rate:           {summary['task_pass_rate']}")
        print(f"\nTotal Test Cases:         {summary['total_test_cases']}")
        print(f"Passed Test Cases:        {summary['total_passed_tests']}")
        print(f"Failed Test Cases:        {summary['total_failed_tests']}")
        print(f"Test Case Pass Rate:      {summary['test_case_pass_rate']}")
        print("="*80 + "\n")
    
    def save_to_json(self, output_file: str = "unit_test_summary.json"):
        """
        Save results to JSON file.
        
        Args:
            output_file: Output JSON file path
        """
        output_path = Path(output_file)
        
        summary = self.generate_summary()
        
        output_data = {
            'summary': summary,
            'per_task_results': self.results
        }
        
        with open(output_path, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        print(f"Results saved to: {output_path.absolute()}")
    
    def save_to_csv(self, output_file: str = "unit_test_summary.csv"):
        """
        Save results to CSV file.
        
        Args:
            output_file: Output CSV file path
        """
        import csv
        
        output_path = Path(output_file)
        
        with open(output_path, 'w', newline='') as f:
            fieldnames = ['task_id', 'status', 'total_tests', 'passed_tests', 
                         'failed_tests', 'final_line']
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            
            writer.writeheader()
            for result in self.results:
                row = {k: result.get(k, '') for k in fieldnames}
                writer.writerow(row)
        
        print(f"Results saved to: {output_path.absolute()}")


def main():
    """
    Main entry point for the script.
    """
    # Initialize summarizer
    summarizer = UnitTestSummary()
    
    # Scan all tasks
    print("Scanning task folders...")
    summarizer.scan_all_tasks()
    
    # Print results
    summarizer.print_results()
    
    # Save to files
    summarizer.save_to_json("./unit_test_summary.json")
    summarizer.save_to_csv("./unit_test_summary.csv")


if __name__ == "__main__":
    main()