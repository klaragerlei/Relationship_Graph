import unittest
import sys
import pandas as pd
from app.create_relationship_graph import round_time


class TestBasicSetup(unittest.TestCase):

    def test_python_version(self):
        """Just verify Python is working"""
        self.assertGreater(sys.version_info.major, 2)


class TestModuleImport(unittest.TestCase):

    def test_can_import_round_time(self):
        """Test if we can import one simple function"""
        self.assertTrue(callable(round_time))


class TestRoundTime(unittest.TestCase):

    def test_round_time_to_hour(self):
        """Test the round_time function"""
        # prepare
        dates = pd.Series(pd.to_datetime(['2020-01-01 10:23:45', '2020-01-01 11:47:12']))

        # execute
        actual = round_time(dates, 'h')

        # verify - use simple assertions instead of pd.testing
        self.assertEqual(len(actual), 2)
        self.assertEqual(actual.iloc[0], pd.Timestamp('2020-01-01 10:00:00'))
        self.assertEqual(actual.iloc[1], pd.Timestamp('2020-01-01 11:00:00'))


if __name__ == '__main__':
    print(f"Python version: {sys.version}")
    unittest.main(verbosity=2)