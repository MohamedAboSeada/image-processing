import unittest
import sys
import os

if __name__ == '__main__':
    # Add the current directory to the path
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    
    # Discover and run all tests
    test_loader = unittest.TestLoader()
    test_suite = test_loader.discover('tests', pattern='test_*.py')
    
    # Run the tests
    test_runner = unittest.TextTestRunner(verbosity=2)
    result = test_runner.run(test_suite)
    
    # Exit with non-zero code if tests failed
    sys.exit(not result.wasSuccessful())
