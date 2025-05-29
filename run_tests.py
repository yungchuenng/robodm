#!/usr/bin/env python3
"""Test runner script for fog_x tests."""

import subprocess
import sys
import argparse
import os


def run_command(cmd):
    """Run a command and return the result."""
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    print(result.stdout)
    if result.stderr:
        print(result.stderr, file=sys.stderr)
    return result.returncode


def main():
    parser = argparse.ArgumentParser(description="Run fog_x tests")
    parser.add_argument(
        "--test-type", 
        choices=["unit", "integration", "benchmark", "all"], 
        default="unit",
        help="Type of tests to run"
    )
    parser.add_argument(
        "--verbose", "-v", 
        action="store_true", 
        help="Verbose output"
    )
    parser.add_argument(
        "--coverage", "-c", 
        action="store_true", 
        help="Run with coverage reporting"
    )
    parser.add_argument(
        "--benchmark-size", 
        choices=["small", "medium", "large"], 
        default="small",
        help="Size of benchmark datasets"
    )
    parser.add_argument(
        "--output-dir", 
        default="./test_results",
        help="Directory for test outputs"
    )
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Base pytest command
    cmd = ["python", "-m", "pytest"]
    
    if args.verbose:
        cmd.extend(["-v", "-s"])
    
    if args.coverage:
        cmd.extend([
            "--cov=fog_x", 
            "--cov-report=html:" + os.path.join(args.output_dir, "coverage"),
            "--cov-report=term"
        ])
    
    # Test selection based on type
    if args.test_type == "unit":
        cmd.extend([
            "tests/test_trajectory.py", 
            "tests/test_loaders.py",
            "-m", "not slow and not integration and not benchmark"
        ])
    elif args.test_type == "integration":
        cmd.extend([
            "tests/test_trajectory.py::TestTrajectoryIntegration",
            "tests/test_loaders.py::TestLoaderComparison",
            "-m", "integration"
        ])
    elif args.test_type == "benchmark":
        cmd.extend([
            "tests/test_benchmark.py",
            "-m", "benchmark"
        ])
        if args.benchmark_size == "large":
            cmd.extend(["--benchmark-size=large"])
        elif args.benchmark_size == "medium":
            cmd.extend(["-m", "not slow"])
    elif args.test_type == "all":
        cmd.extend(["tests/"])
    
    # Add output options
    cmd.extend([
        "--junitxml=" + os.path.join(args.output_dir, "test_results.xml"),
        "--html=" + os.path.join(args.output_dir, "test_report.html"),
        "--self-contained-html"
    ])
    
    # Run tests
    return_code = run_command(cmd)
    
    if return_code == 0:
        print("\n✅ All tests passed!")
        if args.test_type == "benchmark":
            print(f"📊 Benchmark results saved to {args.output_dir}")
    else:
        print("\n❌ Some tests failed!")
        print(f"📋 Detailed results available in {args.output_dir}")
    
    return return_code


if __name__ == "__main__":
    sys.exit(main()) 