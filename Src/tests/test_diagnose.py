
import pytest
import subprocess
import os
import sys

def test_diagnose_script_runs():
    """Verify that the diagnose.py script can be executed."""
    # Add Src to path
    env = os.environ.copy()
    src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    env["PYTHONPATH"] = f"{src_path}:{env.get('PYTHONPATH', '')}"
    
    script_path = os.path.join(src_path, "Main_Scripts", "core", "diagnose.py")
    
    # Run script
    result = subprocess.run(
        [sys.executable, script_path],
        capture_output=True,
        text=True,
        env=env
    )
    
    # It should not crash (exit code 0)
    # Note: It might print errors if CUDA is missing, but the script itself should handle them gracefully.
    assert result.returncode == 0
    # Check for the actual header printed by the new diagnose.py
    assert "PyTorch CUDA Profiler" in result.stdout
