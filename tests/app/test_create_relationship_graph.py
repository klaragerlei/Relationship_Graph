import unittest
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib

matplotlib.use('Agg')  # Use non-interactive backend for testing
import matplotlib.pyplot as plt

# Import functions to test
try:
    from app.create_relationship_graph import (
        round_time,
        remove_duplicate_times,
        align_trajectories,
        count_close_encounters,
        build_graph_from_samples,
        get_edge_weights,
        calculate_edge_threshold,
        filter_edges_by_threshold,
        build_filtered_graph,
        normalize_edge_widths,
        filter_widths_by_threshold,
        create_color_map,
        assign_node_colors
    )
except ImportError as e:
    print(f"Import error: {e}")
    raise


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

    def test_no_duplicates(self):
        # prepare
        index = pd.DatetimeIndex(['2020-01-01 10:00:00', '2020-01-01 11:00:00'])
        df = pd.DataFrame({'x': [1, 2], 'y': [3, 4]}, index=index)

        # execute
        actual = remove_duplicate_times(df)

        # verify
        self.assertEqual(len(actual), 2)


class TestAlignTrajectories(unittest.TestCase):

    def test_align_trajectories(self):
        # prepare
        index_a = pd.DatetimeIndex(['2020-01-01 10:00:00', '2020-01-01 11:00:00'])
        index_b = pd.DatetimeIndex(['2020-01-01 10:00:00', '2020-01-01 12:00:00'])
        traj_a = pd.DataFrame({'x': [1, 2], 'y': [3, 4]}, index=index_a)
        traj_b = pd.DataFrame({'x': [5, 6], 'y': [7, 8]}, index=index_b)

        # execute
        aligned_a, aligned_b = align_trajectories(traj_a, traj_b)

        # verify
        self.assertEqual(len(aligned_a), 3)
        self.assertEqual(len(aligned_b), 3)


class TestCountCloseEncounters(unittest.TestCase):

    def test_count_close_encounters(self):
        # prepare
        distances = pd.Series([10, 15, 25, 30, 5])
        meeting_distance = 20

        # execute
        actual = count_close_encounters(distances, meeting_distance)

        # verify
        self.assertEqual(actual, 3)

    def test_count_close_encounters_none_close(self):
        # prepare
        distances = pd.Series([50, 60, 70])
        meeting_distance = 20

        # execute
        actual = count_close_encounters(distances, meeting_distance)

        # verify
        self.assertEqual(actual, 0)


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

    def test_build_graph_single_edge(self):
        # prepare
        samples = {('A', 'B'): 100}

        # execute
        actual = build_graph_from_samples(samples)

        # verify
        self.assertEqual(len(actual.nodes()), 2)
        self.assertEqual(len(actual.edges()), 1)


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

    def test_calculate_edge_threshold_median(self):
        # prepare
        weight_values = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        percentile = 50

        # execute
        actual = calculate_edge_threshold(weight_values, percentile)

        # verify
        self.assertEqual(actual, 5.5)

    def test_calculate_edge_threshold_high(self):
        # prepare
        weight_values = [1, 2, 3, 4, 5]
        percentile = 80

        # execute
        actual = calculate_edge_threshold(weight_values, percentile)

        # verify
        self.assertGreater(actual, 3)


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

    def test_filter_edges_all_pass(self):
        # prepare
        widths = {('A', 'B'): 10, ('B', 'C'): 20}
        threshold = 5

        # execute
        actual = filter_edges_by_threshold(widths, threshold)

        # verify
        self.assertEqual(len(actual), 2)


class TestBuildFilteredGraph(unittest.TestCase):

    def test_build_filtered_graph(self):
        # prepare
        widths = {('A', 'B'): 5, ('B', 'C'): 10, ('C', 'D'): 3}
        threshold = 4

        # execute
        actual = build_filtered_graph(widths, threshold)

        # verify
        self.assertEqual(len(actual.edges()), 2)
        self.assertTrue(actual.has_edge('A', 'B'))
        self.assertTrue(actual.has_edge('B', 'C'))


class TestNormalizeEdgeWidths(unittest.TestCase):

    def test_normalize_edge_widths(self):
        # prepare
        weight_values = [1, 5, 10, 15, 20]
        min_edge_width = 0.5
        max_edge_width = 3.5

        # execute
        actual = normalize_edge_widths(weight_values, min_edge_width, max_edge_width)

        # verify
        self.assertEqual(len(actual), 5)
        for width in actual:
            self.assertGreaterEqual(width, min_edge_width - 0.1)
            self.assertLessEqual(width, max_edge_width + 0.1)


class TestFilterWidthsByThreshold(unittest.TestCase):

    def test_filter_widths_by_threshold(self):
        # prepare
        normalized_widths = [1.0, 2.0, 3.0]
        widths = {('A', 'B'): 5, ('B', 'C'): 10, ('C', 'D'): 3}
        threshold = 4

        # execute
        actual = filter_widths_by_threshold(normalized_widths, widths, threshold)

        # verify
        self.assertEqual(len(actual), 2)


class TestCreateColorMap(unittest.TestCase):

    def test_create_color_map(self):
        # prepare
        unique_groups = ['GroupA', 'GroupB', 'GroupC']

        # execute
        actual = create_color_map(unique_groups)

        # verify
        self.assertEqual(len(actual), 3)
        self.assertIn('GroupA', actual)
        for color in actual.values():
            self.assertTrue(color.startswith('#'))

    def test_create_color_map_single_group(self):
        # prepare
        unique_groups = ['OnlyGroup']

        # execute
        actual = create_color_map(unique_groups)

        # verify
        self.assertEqual(len(actual), 1)


class TestAssignNodeColors(unittest.TestCase):

    def test_assign_node_colors(self):
        # prepare
        graph = nx.Graph()
        graph.add_edge('A', 'B')
        graph.add_edge('B', 'C')
        node_to_group = {'A': 'Group1', 'B': 'Group1', 'C': 'Group2'}
        color_map = {'Group1': '#FF0000', 'Group2': '#00FF00'}

        # execute
        actual = assign_node_colors(graph, node_to_group, color_map)

        # verify
        self.assertEqual(len(actual), 3)
        self.assertEqual(actual[0], '#FF0000')
        self.assertEqual(actual[2], '#00FF00')

    def test_assign_node_colors_missing_node(self):
        # prepare
        graph = nx.Graph()
        graph.add_edge('A', 'B')
        node_to_group = {'A': 'Group1'}
        color_map = {'Group1': '#FF0000'}

        # execute
        actual = assign_node_colors(graph, node_to_group, color_map)

        # verify
        self.assertEqual(len(actual), 2)
        self.assertEqual(actual[0], '#FF0000')
        self.assertEqual(actual[1], 'lightgrey')


if __name__ == '__main__':
    unittest.main()