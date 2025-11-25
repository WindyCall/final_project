"""
Extract unit test cases and results from test-stdout.txt files in data/raw/inference_result.
Updates both unit_test_summary.csv and unit_test_summary.json with detailed test information.
"""

import os
import re
import csv
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional


def parse_test_stdout(file_path: str) -> Optional[Dict]:
    """
    Parse a test-stdout.txt file to extract test results and raw test output.

    Args:
        file_path: Path to the test-stdout.txt file

    Returns:
        Dictionary containing parsed test information, or None if parsing fails
    """
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return None

    # Find the end of solution marker
    end_marker = "======== END OF SOLUTION.PY CONTENT ========"
    if end_marker not in content:
        print(f"Warning: No solution end marker found in {file_path}")
        return None

    # Extract test results section (after the end marker)
    test_section = content.split(end_marker)[1].strip()

    # Parse final statistics
    stats_pattern = re.compile(r'=+ (\d+) passed.*in ([\d.]+)s =+')
    stats_match = stats_pattern.search(test_section)

    passed_count = 0
    total_count = 0
    test_time = 0.0

    if stats_match:
        passed_count = int(stats_match.group(1))
        test_time = float(stats_match.group(2))
        total_count = passed_count

    # Check for failed tests
    failed_pattern = re.compile(r'(\d+) failed.*?(\d+) passed')
    failed_match = failed_pattern.search(test_section)

    failed_count = 0
    if failed_match:
        failed_count = int(failed_match.group(1))
        passed_count = int(failed_match.group(2))
        total_count = failed_count + passed_count

    # Extract final status line
    final_line = ""
    if "🎉 TASK PASSED" in test_section:
        final_match = re.search(r'🎉 TASK PASSED: (.+)', test_section)
        if final_match:
            final_line = f"🎉 TASK PASSED: {final_match.group(1)}"
            status = "PASSED"
    elif "❌ TASK FAILED" in test_section:
        final_match = re.search(r'❌ TASK FAILED: (.+)', test_section)
        if final_match:
            final_line = f"❌ TASK FAILED: {final_match.group(1)}"
            status = "FAILED"
    else:
        status = "UNKNOWN"
        final_line = "No final status found"

    return {
        'raw_test_output': test_section,
        'total_tests': total_count,
        'passed_tests': passed_count,
        'failed_tests': failed_count,
        'test_time': test_time,
        'status': status,
        'final_line': final_line
    }


def extract_all_unit_tests(inference_dir: str) -> Dict[str, Dict]:
    """
    Extract unit test information from all task directories.

    Args:
        inference_dir: Path to the inference_result directory

    Returns:
        Dictionary mapping task_id to test information
    """
    results = {}
    inference_path = Path(inference_dir)

    if not inference_path.exists():
        print(f"Error: Directory {inference_dir} does not exist")
        return results

    # Iterate through all task directories
    task_dirs = sorted([d for d in inference_path.iterdir() if d.is_dir()])

    print(f"Found {len(task_dirs)} task directories")

    for task_dir in task_dirs:
        task_id = task_dir.name
        test_file = task_dir / "verifier" / "test-stdout.txt"

        if not test_file.exists():
            print(f"Warning: test-stdout.txt not found for {task_id}")
            continue

        print(f"Processing {task_id}...")
        test_data = parse_test_stdout(str(test_file))

        if test_data:
            results[task_id] = test_data
        else:
            print(f"Warning: Failed to parse test data for {task_id}")

    return results


def update_csv_summary(results: Dict[str, Dict], output_file: str):
    """
    Update the unit_test_summary.csv file with extracted data.

    Args:
        results: Dictionary of test results by task_id
        output_file: Path to the output CSV file
    """
    fieldnames = ['task_id', 'status', 'total_tests', 'passed_tests', 'failed_tests', 'final_line', 'raw_test_output']

    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for task_id in sorted(results.keys()):
            data = results[task_id]
            writer.writerow({
                'task_id': task_id,
                'status': data['status'],
                'total_tests': data['total_tests'],
                'passed_tests': data['passed_tests'],
                'failed_tests': data['failed_tests'],
                'final_line': data['final_line'],
                'raw_test_output': data['raw_test_output']
            })

    print(f"CSV summary written to {output_file}")


def update_json_summary(results: Dict[str, Dict], output_file: str):
    """
    Update the unit_test_summary.json file with detailed test data.

    Args:
        results: Dictionary of test results by task_id
        output_file: Path to the output JSON file
    """
    # Convert to a format suitable for JSON
    json_data = {}

    for task_id, data in sorted(results.items()):
        json_data[task_id] = {
            'status': data['status'],
            'total_tests': data['total_tests'],
            'passed_tests': data['passed_tests'],
            'failed_tests': data['failed_tests'],
            'test_time': data['test_time'],
            'final_line': data['final_line'],
            'raw_test_output': data['raw_test_output']
        }

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(json_data, f, indent=2, ensure_ascii=False)

    print(f"JSON summary written to {output_file}")


def main():
    """Main function to extract and update unit test summaries."""
    # Set up paths
    project_root = Path(__file__).parent.parent.parent
    inference_dir = project_root / "data" / "raw" / "inference_result"
    output_dir = project_root / "outputs" / "results"

    csv_output = output_dir / "unit_test_summary.csv"
    json_output = output_dir / "unit_test_summary.json"

    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)

    # Extract all unit test data
    print("Extracting unit test data from all tasks...")
    results = extract_all_unit_tests(str(inference_dir))

    print(f"\nSuccessfully extracted data from {len(results)} tasks")

    # Update CSV summary
    print("\nUpdating CSV summary...")
    update_csv_summary(results, str(csv_output))

    # Update JSON summary
    print("\nUpdating JSON summary...")
    update_json_summary(results, str(json_output))

    # Print statistics
    total_tasks = len(results)
    passed_tasks = sum(1 for r in results.values() if r['status'] == 'PASSED')
    failed_tasks = sum(1 for r in results.values() if r['status'] == 'FAILED')

    print("\n" + "="*60)
    print("Summary Statistics:")
    print("="*60)
    print(f"Total tasks processed: {total_tasks}")
    print(f"Tasks passed: {passed_tasks} ({passed_tasks/total_tasks*100:.1f}%)")
    print(f"Tasks failed: {failed_tasks} ({failed_tasks/total_tasks*100:.1f}%)")
    print("="*60)


if __name__ == "__main__":
    main()
