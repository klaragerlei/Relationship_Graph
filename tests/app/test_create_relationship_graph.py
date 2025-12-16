import unittest
import pandas as pd
import numpy as np
import networkx as nx

from app.create_relationship_graph import (
    round_time,
    remove_duplicate_times,
    count_close_encounters,
    build_graph_from_samples,
    get_edge_weights,
    calculate_edge_threshold,
    filter_edges_by_threshold
)


class TestRoundTime(unittest.TestCase):

    def test_round_time_to_hour(self):
        # prepare
        dates = pd.Series(pd.to_datetime(['2020-01-01 10:23:45', '2020-01-01 11:47:12']))

        # execute
        actual = round_time(dates, 'h')

        # verify
        expected = pd.Series(pd.to_datetime(['2020-01-01 10:00:00', '2020-01-01 11:00:00']))
        pd.testing.assert_series_equal(actual, expected)


class TestRemoveDuplicateTimes(unittest.TestCase):

    def test_remove_duplicate_times(self):
        # prepare
        index = pd.DatetimeIndex(['2020-01-01 10:00:00', '2020-01-01 10:00:00', '2020-01-01 11:00:00'])
        df = pd.DataFrame({'x': [1, 2, 3], 'y': [4, 5, 6]}, index=index)

        # execute
        actual = remove_duplicate_times(df)

        # verify
        self.assertEqual(len(actual), 2)
        self.assertEqual(actual.iloc[0]['x'], 1)


class TestCountCloseEncounters(unittest.TestCase):

    def test_count_close_encounters(self):
        # prepare
        distances = pd.Series([10, 15, 25, 30, 5])
        meeting_distance = 20

        # execute
        actual = count_close_encounters(distances, meeting_distance)

        # verify
        self.assertEqual(actual, 3)


class TestBuildGraphFromSamples(unittest.TestCase):

    def test_build_graph_from_samples(self):
        # prepare
        samples = {
            ('ID_1', 'ID_2'): 5,
            ('ID_1', 'ID_3'): 10
        }

        # execute
        actual = build_graph_from_samples(samples)

        # verify
        self.assertEqual(len(actual.nodes()), 3)
        self.assertEqual(len(actual.edges()), 2)
        self.assertEqual(actual['ID_1']['ID_2']['weight'], 5)


class TestGetEdgeWeights(unittest.TestCase):

    def test_get_edge_weights(self):
        # prepare
        graph = nx.Graph()
        graph.add_edge('A', 'B', weight=5)
        graph.add_edge('B', 'C', weight=10)

        # execute
        actual = get_edge_weights(graph)

        # verify
        self.assertEqual(len(actual), 2)
        self.assertIn(('A', 'B'), actual)


class TestCalculateEdgeThreshold(unittest.TestCase):

    def test_calculate_edge_threshold(self):
        # prepare
        weight_values = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        percentile = 50

        # execute
        actual = calculate_edge_threshold(weight_values, percentile)

        # verify
        self.assertEqual(actual, 5.5)


class TestFilterEdgesByThreshold(unittest.TestCase):

    def test_filter_edges_by_threshold(self):
        # prepare
        widths = {('A', 'B'): 5, ('B', 'C'): 10, ('C', 'D'): 3}
        threshold = 4

        # execute
        actual = filter_edges_by_threshold(widths, threshold)

        # verify
        self.assertEqual(len(actual), 2)
        self.assertIn(('A', 'B'), actual)
        self.assertIn(('B', 'C'), actual)


if __name__ == '__main__':
    unittest.main()