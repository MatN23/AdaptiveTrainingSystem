import sys
import subprocess
import argparse
from pathlib import Path
import time


def run_command(cmd, description):
    """Run a command and report results."""
    print(f"\\n{'='*80}")
    print(f"Running: {description}")
    print(f"{'='*80}")
    print(f"Command: {' '.join(cmd)}")
    
    start_time = time.time()
    result = subprocess.run(cmd, capture_output=False)
    elapsed = time.time() - start_time
    
    print(f"\\nCompleted in {elapsed:.2f}s")
    
    return result.returncode == 0


def main():
    parser = argparse.ArgumentParser(description="Run comprehensive tests")
    parser.add_argument('--fast', action='store_true', help='Skip slow tests')
    parser.add_argument('--model-only', action='store_true', help='Only run model tests')
    parser.add_argument('--trainer-only', action='store_true', help='Only run trainer tests')
    parser.add_argument('--integration', action='store_true', help='Only run integration tests')
    parser.add_argument('--performance', action='store_true', help='Only run performance tests')
    parser.add_argument('--coverage', action='store_true', help='Generate coverage report')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    parser.add_argument('--gpu', action='store_true', help='Include GPU tests')
    
    args = parser.parse_args()
    
    # Determine which tests to run
    test_files = []
    
    # Get the tests directory path
    tests_dir = Path(__file__).parent
    
    if args.model_only:
        test_files = [str(tests_dir / 'test_model.py')]
    elif args.trainer_only:
        test_files = [str(tests_dir / 'test_trainer.py')]
    elif args.integration:
        test_files = [str(tests_dir / 'test_integration.py')]
    elif args.performance:
        test_files = [str(tests_dir / 'test_performance.py')]
    else:
        test_files = [
            str(tests_dir / 'test_model.py'),
            str(tests_dir / 'test_tokenizer.py'),
            str(tests_dir / 'test_dataset.py'),
            str(tests_dir / 'test_trainer.py'),
            str(tests_dir / 'test_integration.py'),
            str(tests_dir / 'test_performance.py'),
            str(tests_dir / 'test_orchestrator.py'),
            str(tests_dir / 'test_initialization.py'),
        ]
        if not args.fast:
            test_files.append(str(tests_dir / 'test_e2e.py'))
    
    # Build pytest command
    pytest_cmd = ['pytest']
    
    if args.verbose:
        pytest_cmd.append('-v')
    else:
        pytest_cmd.append('-q')
    
    if args.fast:
        pytest_cmd.extend(['-m', 'not slow'])
    
    if args.coverage:
        pytest_cmd.extend([
            '--cov=.',
            '--cov-report=html',
            '--cov-report=term-missing'
        ])
    
    if args.gpu:
        pytest_cmd.extend(['--run-gpu-tests'])
    
    pytest_cmd.extend(test_files)
    
    # Run tests
    success = run_command(pytest_cmd, "Pytest Test Suite")
    
    if success:
        print(f"\\n{'='*80}")
        print("✅ ALL TESTS PASSED")
        print(f"{'='*80}")
        
        if args.coverage:
            print(f"\\nCoverage report generated: htmlcov/index.html")
        
        return 0
    else:
        print(f"\\n{'='*80}")
        print("❌ TESTS FAILED")
        print(f"{'='*80}")
        return 1


if __name__ == '__main__':
    sys.exit(main())