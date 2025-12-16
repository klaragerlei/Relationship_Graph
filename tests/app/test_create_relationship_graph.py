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


class TestModuleImport(unittest.TestCase):

    def test_can_import_round_time(self):
        """Test if we can import one simple function"""
        try:
            from app.create_relationship_graph import round_time
            self.assertTrue(callable(round_time))
        except Exception as e:
            self.fail(f"Failed to import: {type(e).__name__}: {str(e)}")


if __name__ == '__main__':
    print(f"Python version: {sys.version}")
    unittest.main(verbosity=2)