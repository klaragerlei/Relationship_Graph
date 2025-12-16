import unittest
import sys


class TestBasicSetup(unittest.TestCase):

    def test_python_version(self):
        """Just verify Python is working"""
        self.assertGreater(sys.version_info.major, 2)

    def test_unittest_works(self):
        """Verify unittest framework is working"""
        self.assertTrue(True)
        self.assertEqual(1 + 1, 2)


if __name__ == '__main__':
    # Print Python version for debugging
    print(f"Python version: {sys.version}")
    print(f"Python path: {sys.path}")

    unittest.main()