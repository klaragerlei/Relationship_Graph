import unittest
import networkx as nx
import numpy as np
from app.create_relationship_graph import (
    build_graph_from_samples,
    get_edge_weights,
    calculate_edge_threshold,
    filter_edges_by_threshold,
    build_filtered_graph,
    create_color_map,
    assign_node_colors
)


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


if __name__ == '__main__':
    unittest.main()