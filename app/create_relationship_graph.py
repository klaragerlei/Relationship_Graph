import logging
from movingpandas import TrajectoryCollection
import geopandas as gpd
import itertools
from datetime import datetime
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd


def round_time(dates: datetime, unit='H'):
    return dates.floor(unit)


def interpolate_points(trajectory: gpd.GeoDataFrame, unit='h'):
    # make new df with time, x and y coord
    trajectory_df = pd.DataFrame({'x': trajectory.geometry.x, 'y': trajectory.geometry.y}, index=trajectory.index)
    # use pandas interpolate
    trajectory_df = trajectory_df.resample(unit).mean()   # this has nans in gaps for now
    return trajectory_df


def calculate_distance_between_trajectories(trajectory_a, trajectory_b):
    # 1: interpolate and resample
    # round (based on param) 'T', 'M' or 'H'
    times_a_rounded_time = round_time(trajectory_a.index, 'h')
    times_b_rounded_time = round_time(trajectory_b.index, 'h')

    trajectory_a.index = times_a_rounded_time
    trajectory_b.index = times_b_rounded_time

    # remove extras (or avg) - there might be multiple identical values after rounding
    times_a_unique = trajectory_a[~trajectory_a.index.duplicated(keep='first')]
    times_b_unique = trajectory_b[~trajectory_b.index.duplicated(keep='first')]

    # interpolate values / or fill with none (depending on gap size if the data is too sparse?)
    times_a_interpolated = interpolate_points(times_a_unique)
    times_b_interpolated = interpolate_points(times_b_unique)

    # 2: match series
    a_aligned, b_aligned = times_a_interpolated.align(times_b_interpolated, join="outer", axis=0)
    # run (1) for both trajectories that are compared and cut off beginnings and ends, so they fully match
    # 3: calculate distance
    a_gdf = gpd.GeoDataFrame(a_aligned, geometry=gpd.points_from_xy(a_aligned['x'], a_aligned['y'])).set_crs(
        'EPSG:4326').to_crs('EPSG:3857')
    b_gdf = gpd.GeoDataFrame(b_aligned, geometry=gpd.points_from_xy(b_aligned['x'], b_aligned['y'])).set_crs(
        'EPSG:4326').to_crs('EPSG:3857')
    distance = a_gdf.distance(b_gdf)  # meters
    return distance


def calculate_distances_between_pairs_of_trajectories(data):
    number_of_animals = len(data.trajectories)
    distances = {}
    for pair in itertools.combinations(range(number_of_animals), 2):
        distance = calculate_distance_between_trajectories(data.trajectories[pair[0]].df, data.trajectories[pair[1]].df)
        distances[pair] = distance
    return distances


def count_number_of_samples_when_close(data, distances, meeting_distance=1000):
    number_of_animals = len(data.trajectories)
    samples_when_close = {}
    for pair in itertools.combinations(range(number_of_animals), 2):
        print(pair)
        distance = distances[pair]  # data frame with distances
        samples_when_close[pair] = (distance < meeting_distance).sum()
    return samples_when_close


def create_graph(data: TrajectoryCollection):
    logging.info("Create relationship graphs")
    distances = calculate_distances_between_pairs_of_trajectories(data)
    samples_when_close = count_number_of_samples_when_close(data, distances)
    print(samples_when_close)

    relationship_graph = nx.Graph()
    for key in samples_when_close:
        # add edges
        relationship_graph.add_edge(str(key[0]), str(key[1]), weight=samples_when_close[key])  # todo get animal ID here instead
    pos = nx.spring_layout(relationship_graph, seed=111)

    # nodes
    nx.draw_networkx_nodes(relationship_graph, pos, node_size=700)

    # edges
    nx.draw_networkx_edges(relationship_graph, pos, edgelist=relationship_graph.edges(data=True), width=6)
    nx.draw_networkx_edges(
        relationship_graph, pos, edgelist=relationship_graph.edges(data=True), width=6, alpha=0.5, edge_color="b", style="dashed"
    )

    # node labels
    nx.draw_networkx_labels(relationship_graph, pos, font_size=20, font_family="sans-serif")  # todo make it look nicer
    # edge weight labels
    edge_labels = nx.get_edge_attributes(relationship_graph, "weight")
    nx.draw_networkx_edge_labels(relationship_graph, pos, edge_labels)

    ax = plt.gca()
    ax.margins(0.08)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig('graph.png')  # todo save it properly
    print('')

    # parameterize and define 'close' - for how long and what distance should they be to be considered interacting?
    # find distance segments that meet criteria
    # get number of times the animals met and store
    # make graph
