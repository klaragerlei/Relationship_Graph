import unittest
import pandas as pd
import geopandas as gpd
import numpy as np
import networkx as nx
import movingpandas as mpd
from datetime import datetime
import matplotlib.pyplot as plt
from app.create_relationship_graph import (
    round_time,
    interpolate_points,
    remove_duplicate_times,
    align_trajectories,
    convert_to_geodataframe,
    calculate_distance_between_trajectories,
    calculate_distances_between_pairs_of_trajectories,
    count_close_encounters,
    count_number_of_samples_when_close,
    build_graph_from_samples,
    get_edge_weights,
    calculate_edge_threshold,
    filter_edges_by_threshold,
    build_filtered_graph,
    calculate_graph_layout,
    normalize_edge_widths,
    filter_widths_by_threshold,
    extract_node_groups,
    create_color_map,
    assign_node_colors,
    get_node_colors,
    create_graph
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

    def test_round_time_to_day(self):
        # prepare
        dates = pd.Series(pd.to_datetime(['2020-01-01 10:23:45', '2020-01-02 11:47:12']))

        # execute
        actual = round_time(dates, 'D')

        # verify
        expected = pd.Series(pd.to_datetime(['2020-01-01', '2020-01-02']))
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
        self.assertEqual(actual.iloc[0]['x'], 1)  # keeps first
        self.assertEqual(actual.iloc[1]['x'], 3)

    def test_no_duplicates(self):
        # prepare
        index = pd.DatetimeIndex(['2020-01-01 10:00:00', '2020-01-01 11:00:00'])
        df = pd.DataFrame({'x': [1, 2], 'y': [3, 4]}, index=index)

        # execute
        actual = remove_duplicate_times(df)

        # verify
        self.assertEqual(len(actual), 2)


class TestAlignTrajectories(unittest.TestCase):

    def test_align_trajectories_different_times(self):
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
        self.assertTrue(aligned_a.index.equals(aligned_b.index))


class TestConvertToGeoDataFrame(unittest.TestCase):

    def test_convert_to_geodataframe(self):
        # prepare
        df = pd.DataFrame({'x': [0, 1], 'y': [0, 1]})

        # execute
        actual = convert_to_geodataframe(df)

        # verify
        self.assertIsInstance(actual, gpd.GeoDataFrame)
        self.assertEqual(actual.crs.to_string(), 'EPSG:3857')
        self.assertEqual(len(actual), 2)


class TestCountCloseEncounters(unittest.TestCase):

    def test_count_close_encounters(self):
        # prepare
        distances = pd.Series([10, 15, 25, 30, 5])
        meeting_distance = 20

        # execute
        actual = count_close_encounters(distances, meeting_distance)

        # verify
        self.assertEqual(actual, 3)  # 10, 15, and 5 are below 20

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
        self.assertEqual(actual['ID_1']['ID_3']['weight'], 10)


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
        self.assertEqual(actual[('A', 'B')], 5)


class TestCalculateEdgeThreshold(unittest.TestCase):

    def test_calculate_edge_threshold(self):
        # prepare
        weight_values = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        percentile = 50

        # execute
        actual = calculate_edge_threshold(weight_values, percentile)

        # verify
        self.assertEqual(actual, 5.5)

    def test_calculate_edge_threshold_high_percentile(self):
        # prepare
        weight_values = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        percentile = 90

        # execute
        actual = calculate_edge_threshold(weight_values, percentile)

        # verify
        self.assertEqual(actual, 9.1)


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
        self.assertNotIn(('C', 'D'), actual)


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
        self.assertFalse(actual.has_edge('C', 'D'))


class TestCalculateGraphLayout(unittest.TestCase):

    def test_calculate_graph_layout(self):
        # prepare
        graph = nx.Graph()
        graph.add_edge('A', 'B', weight=5)
        graph.add_edge('B', 'C', weight=10)
        node_spacing = 2.0

        # execute
        actual = calculate_graph_layout(graph, node_spacing)

        # verify
        self.assertEqual(len(actual), 3)
        self.assertIn('A', actual)
        self.assertIn('B', actual)
        self.assertIn('C', actual)
        # Each position should be a tuple of (x, y)
        self.assertEqual(len(actual['A']), 2)


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
        # All values should be between min and max
        for width in actual:
            self.assertGreaterEqual(width, min_edge_width)
            self.assertLessEqual(width, max_edge_width)


class TestFilterWidthsByThreshold(unittest.TestCase):

    def test_filter_widths_by_threshold(self):
        # prepare
        normalized_widths = [1.0, 2.0, 3.0]
        widths = {('A', 'B'): 5, ('B', 'C'): 10, ('C', 'D'): 3}
        threshold = 4

        # execute
        actual = filter_widths_by_threshold(normalized_widths, widths, threshold)

        # verify
        self.assertEqual(len(actual), 2)  # Only A-B and B-C pass threshold


class TestExtractNodeGroups(unittest.TestCase):

    def test_extract_node_groups(self):
        # prepare
        df1 = pd.DataFrame({
            'x': [1, 2],
            'y': [3, 4],
            'group_id': ['GroupA', 'GroupA']
        }, index=pd.DatetimeIndex(['2020-01-01', '2020-01-02']))

        df2 = pd.DataFrame({
            'x': [5, 6],
            'y': [7, 8],
            'group_id': ['GroupB', 'GroupB']
        }, index=pd.DatetimeIndex(['2020-01-01', '2020-01-02']))

        traj_collection = mpd.TrajectoryCollection(
            pd.concat([df1, df2]).assign(track_id=['ID_1', 'ID_1', 'ID_2', 'ID_2']),
            traj_id_col='track_id',
            t='index',
            x='x',
            y='y',
            crs='EPSG:4326'
        )

        graph = nx.Graph()
        graph.add_edge('ID_1', 'ID_2')

        # execute
        actual = extract_node_groups(traj_collection, graph, 'group_id')

        # verify
        self.assertEqual(len(actual), 2)
        self.assertEqual(actual['ID_1'], 'GroupA')
        self.assertEqual(actual['ID_2'], 'GroupB')


class TestCreateColorMap(unittest.TestCase):

    def test_create_color_map(self):
        # prepare
        unique_groups = ['GroupA', 'GroupB', 'GroupC']

        # execute
        actual = create_color_map(unique_groups)

        # verify
        self.assertEqual(len(actual), 3)
        self.assertIn('GroupA', actual)
        self.assertIn('GroupB', actual)
        self.assertIn('GroupC', actual)
        # Check that values are hex colors
        for color in actual.values():
            self.assertTrue(color.startswith('#'))


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
        self.assertEqual(actual[0], '#FF0000')  # A
        self.assertEqual(actual[1], '#FF0000')  # B
        self.assertEqual(actual[2], '#00FF00')  # C


class TestCalculateDistanceBetweenTrajectories(unittest.TestCase):

    def test_calculate_distance_between_trajectories(self):
        # prepare - two trajectories at the same location (distance should be ~0)
        index = pd.DatetimeIndex(['2020-01-01 10:00:00', '2020-01-01 11:00:00'])

        # Create GeoDataFrame with proper geometry
        traj_a = gpd.GeoDataFrame(
            {'x': [0, 0], 'y': [0, 0]},
            geometry=gpd.points_from_xy([0, 0], [0, 0]),
            index=index,
            crs='EPSG:4326'
        )

        traj_b = gpd.GeoDataFrame(
            {'x': [0, 0], 'y': [0, 0]},
            geometry=gpd.points_from_xy([0, 0], [0, 0]),
            index=index,
            crs='EPSG:4326'
        )

        # execute
        actual = calculate_distance_between_trajectories(traj_a, traj_b, time_step='h')

        # verify
        self.assertIsInstance(actual, pd.Series)
        # Distance between same points should be approximately 0
        self.assertAlmostEqual(actual.iloc[0], 0, places=0)


class TestCountNumberOfSamplesWhenClose(unittest.TestCase):

    def test_count_number_of_samples_when_close(self):
        # prepare
        df1 = pd.DataFrame({
            'x': [1, 2],
            'y': [3, 4],
            'track_id': ['ID_1', 'ID_1']
        }, index=pd.DatetimeIndex(['2020-01-01', '2020-01-02']))

        df2 = pd.DataFrame({
            'x': [1, 2],
            'y': [3, 4],
            'track_id': ['ID_2', 'ID_2']
        }, index=pd.DatetimeIndex(['2020-01-01', '2020-01-02']))

        traj_collection = mpd.TrajectoryCollection(
            pd.concat([df1, df2]),
            traj_id_col='track_id',
            t='index',
            x='x',
            y='y',
            crs='EPSG:4326'
        )

        distances = {(0, 1): pd.Series([5, 15, 25])}  # 2 close encounters at threshold 20

        # execute
        actual = count_number_of_samples_when_close(traj_collection, distances, meeting_distance=20)

        # verify
        self.assertEqual(len(actual), 1)
        # Should count 2 encounters (5 and 15 are below 20)
        self.assertEqual(list(actual.values())[0], 2)


class TestCreateGraph(unittest.TestCase):

    def test_create_graph_returns_figure(self):
        # prepare - simple trajectory collection with 2 tracks
        df = pd.DataFrame({
            'timestamp': pd.date_range('2020-01-01', periods=10, freq='h').tolist() * 2,
            'x': [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9] * 2,
            'y': [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9] * 2,
            'track_id': ['ID_1'] * 10 + ['ID_2'] * 10
        })

        traj_collection = mpd.TrajectoryCollection(
            df,
            traj_id_col='track_id',
            t='timestamp',
            x='x',
            y='y',
            crs='EPSG:4326'
        )

        # execute
        actual = create_graph(traj_collection, meeting_distance=10000)  # large threshold

        # verify
        self.assertIsInstance(actual, plt.Figure)
        plt.close(actual)

    def test_create_graph_with_different_time_step(self):
        # prepare
        df = pd.DataFrame({
            'timestamp': pd.date_range('2020-01-01', periods=10, freq='min').tolist() * 2,
            'x': [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9] * 2,
            'y': [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9] * 2,
            'track_id': ['ID_1'] * 10 + ['ID_2'] * 10
        })

        traj_collection = mpd.TrajectoryCollection(
            df,
            traj_id_col='track_id',
            t='timestamp',
            x='x',
            y='y',
            crs='EPSG:4326'
        )

        # execute
        actual = create_graph(traj_collection, time_step='min', meeting_distance=10000)

        # verify
        self.assertIsInstance(actual, plt.Figure)
        plt.close(actual)

    def test_create_graph_with_custom_parameters(self):
        # prepare
        df = pd.DataFrame({
            'timestamp': pd.date_range('2020-01-01', periods=10, freq='h').tolist() * 2,
            'x': [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9] * 2,
            'y': [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9] * 2,
            'track_id': ['ID_1'] * 10 + ['ID_2'] * 10
        })

        traj_collection = mpd.TrajectoryCollection(
            df,
            traj_id_col='track_id',
            t='timestamp',
            x='x',
            y='y',
            crs='EPSG:4326'
        )

        # execute with custom parameters
        actual = create_graph(
            traj_collection,
            meeting_distance=10000,
            edge_threshold_percentile=50,
            node_spacing=3.0,
            node_size=500,
            font_size=10,
            figsize=(10, 8),
            label_strategy='none',
            color_by=None
        )

        # verify
        self.assertIsInstance(actual, plt.Figure)
        plt.close(actual)


if __name__ == '__main__':
    unittest.main()