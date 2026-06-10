# Copyright (c) 2025 MatN23. All rights reserved.

import unittest
import sys
import os

# Add Src/Main_Scripts to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../Main_Scripts')))

from training.orchestrator import VerbosityLevel

class TestInitialization(unittest.TestCase):
    def test_verbosity_level_comparison(self):
        """Test that VerbosityLevel supports comparison operators."""
        self.assertTrue(VerbosityLevel.DEBUG >= VerbosityLevel.NORMAL)
        self.assertTrue(VerbosityLevel.TRACE > VerbosityLevel.DEBUG)
        self.assertTrue(VerbosityLevel.SILENT < VerbosityLevel.MINIMAL)
        self.assertEqual(VerbosityLevel.NORMAL, 2)

    def test_traceback_access(self):
        """Test that traceback is accessible without UnboundLocalError."""
        import traceback
        try:
            raise ValueError("Test error")
        except ValueError:
            # This should not raise UnboundLocalError
            formatted = traceback.format_exc()
            self.assertIn("ValueError: Test error", formatted)

if __name__ == '__main__':
    unittest.main()
