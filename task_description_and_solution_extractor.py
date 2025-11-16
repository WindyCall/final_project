#!/usr/bin/env python3
"""
Extract problem descriptions and code solutions from LiveCodeBench tasks.

For each task, extracts:
1. Problem description from agent/command-0/command.txt
2. Code solution from verifier/test-stdout.txt (between markers)
3. Unit test results (from previous summarizer)

Outputs a structured dataset ready for LLM-as-a-Judge evaluation.
"""

import os
import re
import json
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict


@dataclass
class TaskData:
    """Structured data for a single coding task."""
    task_id: str
    problem_title: str
    problem_description: str
    code_solution: str
    unit_test_status: str  # PASSED or FAILED
    total_tests: int
    passed_tests: int
    failed_tests: int
    test_file_path: str
    problem_file_path: str


class TaskExtractor:
    """Extract problem descriptions and solutions from task folders."""
    
    def __init__(self, project_root: str):
        """
        Initialize the extractor.
        
        Args:
            project_root: Root directory containing task folders
        """
        self.project_root = Path(project_root).resolve()
        self.tasks = []
        self.skipped_with_exception = 0
        self.skipped_missing_files = 0
        self.skipped_parse_error = 0
        print(f"Extracting from: {self.project_root}")
    
    def extract_solution_code(self, test_stdout_content: str) -> Optional[str]:
        """
        Extract code solution from test-stdout.txt.
        
        Args:
            test_stdout_content: Full content of test-stdout.txt
            
        Returns:
            Code solution as string, or None if not found
        """
        # Pattern to match the solution block
        pattern = r'={8,}\s*SOLUTION\.PY CONTENT\s*={8,}(.*?)={8,}\s*END OF SOLUTION\.PY CONTENT\s*={8,}'
        
        match = re.search(pattern, test_stdout_content, re.DOTALL)
        
        if match:
            code = match.group(1).strip()
            return code
        
        return None
    
    def extract_problem_description(self, command_txt_content: str) -> Dict[str, str]:
        """
        Extract problem title and description from command.txt.
        
        Args:
            command_txt_content: Full content of command.txt
            
        Returns:
            Dictionary with 'title' and 'description'
        """
        lines = command_txt_content.strip().split('\n')
        
        # Remove the first line if it's a command (starts with 'claude')
        if lines and lines[0].strip().startswith('claude'):
            lines = lines[1:]
        
        content = '\n'.join(lines).strip()
        
        # Try to extract title (usually in format "Title: X" or just first line)
        title = "Unknown Title"
        title_match = re.search(r'Title:\s*(.+)', content, re.IGNORECASE)
        if title_match:
            title = title_match.group(1).strip()
        else:
            # Use first non-empty line as title
            for line in lines:
                if line.strip() and not line.strip().startswith('==='):
                    title = line.strip()
                    break
        
        return {
            'title': title,
            'description': content
        }
    
    def extract_unit_test_result(self, test_stdout_content: str) -> Dict:
        """
        Extract unit test results from test-stdout.txt.
        
        Returns:
            Dictionary with status, total_tests, passed_tests, failed_tests
        """
        lines = test_stdout_content.strip().split('\n')
        
        # Look for the final result line
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
                'failed_tests': 0
            }
        
        # Pattern 1: "TASK PASSED: All X tests passed!"
        passed_match = re.search(
            r'TASK PASSED:\s*All\s+(\d+)\s+tests?\s+passed', 
            final_line, 
            re.IGNORECASE
        )
        
        if passed_match:
            total = int(passed_match.group(1))
            return {
                'status': 'PASSED',
                'total_tests': total,
                'passed_tests': total,
                'failed_tests': 0
            }
        
        # Pattern 2: "TASK FAILED: X out of Y tests failed!"
        failed_match = re.search(
            r'TASK FAILED:\s*(\d+)\s+out of\s+(\d+)\s+tests?\s+failed', 
            final_line, 
            re.IGNORECASE
        )
        
        if failed_match:
            failed = int(failed_match.group(1))
            total = int(failed_match.group(2))
            return {
                'status': 'FAILED',
                'total_tests': total,
                'passed_tests': total - failed,
                'failed_tests': failed
            }
        
        # Pattern 3: pytest format
        pytest_passed = re.search(r'(\d+)\s+passed', final_line)
        pytest_failed = re.search(r'(\d+)\s+failed', final_line)
        
        if pytest_passed or pytest_failed:
            passed = int(pytest_passed.group(1)) if pytest_passed else 0
            failed = int(pytest_failed.group(1)) if pytest_failed else 0
            total = passed + failed
            
            return {
                'status': 'PASSED' if failed == 0 else 'FAILED',
                'total_tests': total,
                'passed_tests': passed,
                'failed_tests': failed
            }
        
        return {
            'status': 'UNKNOWN',
            'total_tests': 0,
            'passed_tests': 0,
            'failed_tests': 0
        }
    
    def process_task(self, task_folder: Path) -> Optional[TaskData]:
        """
        Process a single task folder and extract all relevant data.
        
        Args:
            task_folder: Path to task folder
            
        Returns:
            TaskData object or None if extraction failed
        """
        task_id = task_folder.name
        
        # Skip tasks with exception.txt (execution errors)
        exception_file = task_folder / "exception.txt"
        if exception_file.exists():
            print(f"⊘ {task_id}: Skipped (has exception.txt)")
            self.skipped_with_exception += 1
            return None
        
        # Find paths to required files
        test_stdout_paths = [
            task_folder / "verifier" / "test-stdout.txt",
            task_folder / "verifier" / "verifier" / "test-stdout.txt",
            task_folder / "test-stdout.txt",
        ]
        
        command_txt_paths = [
            task_folder / "agent" / "command-0" / "command.txt",
            task_folder / "command-0" / "command.txt",
        ]
        
        # Find test-stdout.txt
        test_file = None
        for path in test_stdout_paths:
            if path.exists():
                test_file = path
                break
        
        if not test_file:
            print(f"⚠️  {task_id}: test-stdout.txt not found")
            self.skipped_missing_files += 1
            return None
        
        # Find command.txt
        command_file = None
        for path in command_txt_paths:
            if path.exists():
                command_file = path
                break
        
        if not command_file:
            print(f"⚠️  {task_id}: command.txt not found")
            self.skipped_missing_files += 1
            return None
        
        try:
            # Read files
            test_content = test_file.read_text(encoding='utf-8', errors='ignore')
            command_content = command_file.read_text(encoding='utf-8', errors='ignore')
            
            # Extract data
            code_solution = self.extract_solution_code(test_content)
            if not code_solution:
                print(f"⚠️  {task_id}: Could not extract code solution")
                self.skipped_parse_error += 1
                return None
            
            problem_info = self.extract_problem_description(command_content)
            test_result = self.extract_unit_test_result(test_content)
            
            # Create TaskData object
            task_data = TaskData(
                task_id=task_id,
                problem_title=problem_info['title'],
                problem_description=problem_info['description'],
                code_solution=code_solution,
                unit_test_status=test_result['status'],
                total_tests=test_result['total_tests'],
                passed_tests=test_result['passed_tests'],
                failed_tests=test_result['failed_tests'],
                test_file_path=str(test_file.relative_to(self.project_root)),
                problem_file_path=str(command_file.relative_to(self.project_root))
            )
            
            print(f"✓ {task_id}: Extracted successfully ({test_result['status']})")
            return task_data
            
        except Exception as e:
            print(f"✗ {task_id}: Error - {str(e)}")
            self.skipped_parse_error += 1
            return None
    
    def scan_all_tasks(self):
        """Scan all task folders and extract data."""
        # Find task folders
        task_folders = []
        
        for item in self.project_root.rglob("*"):
            if item.is_dir() and '_' in item.name:
                # Check if it has required files
                has_test = any((item / p).exists() for p in [
                    "verifier/test-stdout.txt",
                    "verifier/verifier/test-stdout.txt",
                    "test-stdout.txt"
                ])
                has_command = any((item / p).exists() for p in [
                    "agent/command-0/command.txt",
                    "command-0/command.txt"
                ])
                
                if has_test and has_command and item not in task_folders:
                    task_folders.append(item)
        
        print(f"\nFound {len(task_folders)} task folders to process\n")
        print("="*80)
        
        # Process each task
        for task_folder in sorted(task_folders):
            task_data = self.process_task(task_folder)
            if task_data:
                self.tasks.append(task_data)
        
        print("="*80)
        print(f"\n✓ Successfully extracted {len(self.tasks)} tasks")
        print(f"⊘ Skipped {self.skipped_with_exception} tasks with exceptions")
        print(f"⚠️  Skipped {self.skipped_missing_files} tasks with missing files")
        print(f"✗ Skipped {self.skipped_parse_error} tasks with parse errors")
        total_skipped = self.skipped_with_exception + self.skipped_missing_files + self.skipped_parse_error
        print(f"\nTotal: {len(self.tasks)} extracted, {total_skipped} skipped\n")
    
    def get_summary_stats(self) -> Dict:
        """Get summary statistics of extracted tasks."""
        total = len(self.tasks)
        passed = sum(1 for t in self.tasks if t.unit_test_status == 'PASSED')
        failed = sum(1 for t in self.tasks if t.unit_test_status == 'FAILED')
        
        return {
            'total_tasks': total,
            'passed_tasks': passed,
            'failed_tasks': failed,
            'pass_rate': f"{passed}/{total} ({100*passed/total:.1f}%)" if total > 0 else "N/A"
        }
    
    def save_to_json(self, output_file: str = "extracted_tasks.json"):
        """Save extracted data to JSON file."""
        output_path = Path(output_file)
        
        # Convert to dict
        data = {
            'summary': self.get_summary_stats(),
            'skip_statistics': {
                'skipped_with_exception': self.skipped_with_exception,
                'skipped_missing_files': self.skipped_missing_files,
                'skipped_parse_error': self.skipped_parse_error,
                'total_skipped': (self.skipped_with_exception + 
                                 self.skipped_missing_files + 
                                 self.skipped_parse_error)
            },
            'tasks': [asdict(task) for task in self.tasks]
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        print(f"✓ Saved to: {output_path.absolute()}")
    
    def save_separate_files(self, output_dir: str = "extracted_tasks"):
        """
        Save each task to a separate JSON file for easier processing.
        
        Creates structure:
        extracted_tasks/
        ├── 1873_d__ewd5mxi.json
        ├── 2808_BQGEqJL.json
        └── ...
        """
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        for task in self.tasks:
            task_file = output_path / f"{task.task_id}.json"
            with open(task_file, 'w', encoding='utf-8') as f:
                json.dump(asdict(task), f, indent=2, ensure_ascii=False)
        
        print(f"✓ Saved {len(self.tasks)} individual files to: {output_path.absolute()}")
    
    def print_sample(self, n: int = 2):
        """Print sample of extracted data."""
        print("\n" + "="*80)
        print(f"SAMPLE EXTRACTED TASKS (showing {min(n, len(self.tasks))} of {len(self.tasks)})")
        print("="*80)
        
        for i, task in enumerate(self.tasks[:n]):
            print(f"\n{'─'*80}")
            print(f"Task {i+1}: {task.task_id}")
            print(f"{'─'*80}")
            print(f"Title: {task.problem_title}")
            print(f"Status: {task.unit_test_status} ({task.passed_tests}/{task.total_tests} tests)")
            print(f"\nProblem (first 200 chars):")
            print(task.problem_description[:200] + "...")
            print(f"\nCode Solution (first 300 chars):")
            print(task.code_solution[:300] + "...")
            print()


def main():
    """Main entry point."""
    project_path = "./2025-11-04__15-32-48"
    
    # Extract all tasks
    extractor = TaskExtractor(project_path)
    extractor.scan_all_tasks()
    
    # Print summary
    summary = extractor.get_summary_stats()
    print("="*80)
    print("EXTRACTION SUMMARY")
    print("="*80)
    print(f"Total Tasks Extracted:  {summary['total_tasks']}")
    print(f"Passed Tasks:           {summary['passed_tasks']}")
    print(f"Failed Tasks:           {summary['failed_tasks']}")
    print(f"Pass Rate:              {summary['pass_rate']}")
    print(f"\nSkipped Tasks:")
    print(f"  With exceptions:      {extractor.skipped_with_exception}")
    print(f"  Missing files:        {extractor.skipped_missing_files}")
    print(f"  Parse errors:         {extractor.skipped_parse_error}")
    total_skipped = (extractor.skipped_with_exception + 
                     extractor.skipped_missing_files + 
                     extractor.skipped_parse_error)
    print(f"  Total skipped:        {total_skipped}")
    print("="*80 + "\n")
    
    # Print samples
    extractor.print_sample(n=2)
    
    # Save outputs
    output_dir = Path(project_path)
    extractor.save_to_json(output_dir / "extracted_tasks.json")
    extractor.save_separate_files(output_dir / "extracted_tasks")
    
    print("\n" + "="*80)
    print("NEXT STEPS")
    print("="*80)
    print("1. Review extracted_tasks.json for the full dataset")
    print("2. Individual task files are in extracted_tasks/ folder")
    print("3. Use this data for LLM-as-a-Judge evaluation")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()