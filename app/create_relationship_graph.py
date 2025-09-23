import logging
from movingpandas import TrajectoryCollection
import geopandas as gpd
import itertools
from datetime import datetime
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd


def round_time(dates: pd.Series, unit='H'):
    """
    Round series of dates
    """
    return dates.floor(unit)


def interpolate_points(trajectory: gpd.GeoDataFrame, time_unit='h') -> pd.DataFrame:
    """
    todo: should there be an option for smaller gaps to be interpolated?
    Creates data frame with time and x and y coordinates and resamples time series data

    Parameters:
        trajectory (gpd.GeoDataFrame): trajectory data
        time_unit (str): unit of time for resampling
    Returns:
        trajectory_df (pd.DataFrame): resampled trajectory
    """
    # make new df with time, x and y coord
    trajectory_df = pd.DataFrame({'x': trajectory.geometry.x, 'y': trajectory.geometry.y}, index=trajectory.index)
    # resample
    trajectory_df = trajectory_df.resample(time_unit).mean()   # this has nans in gaps
    return trajectory_df


def calculate_distance_between_trajectories(trajectory_a: pd.DataFrame, trajectory_b: pd.DataFrame, time_step='h') -> pd.Series:
    """
    Calculates the distance between two trajectories after aligning them

    Parameters:
        trajectory_a (pd.DataFrame): data frame with trajectory
        trajectory_b (pd.DataFrame): data frame with trajectory
        time_step (str): unit of time used for interpolation ('h' by default)
    Returns:
        distance (pd.Series): distance between trajectory a and b at each time point
    """
    # interpolate and resample
    # round (based on param)
    times_a_rounded_time = round_time(trajectory_a.index, time_step)
    times_b_rounded_time = round_time(trajectory_b.index, time_step)

    trajectory_a.index = times_a_rounded_time
    trajectory_b.index = times_b_rounded_time

    # remove extras (or avg) - there might be multiple identical values after rounding
    times_a_unique = trajectory_a[~trajectory_a.index.duplicated(keep='first')]
    times_b_unique = trajectory_b[~trajectory_b.index.duplicated(keep='first')]

    # interpolate values / or fill with none
    times_a_interpolated = interpolate_points(times_a_unique, time_unit=time_step)
    times_b_interpolated = interpolate_points(times_b_unique, time_unit=time_step)

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


def calculate_distances_between_pairs_of_trajectories(data: TrajectoryCollection) -> dict:
    """
    Calculates the distance between pairs of trajectories for all trajectories in the data.

    Parameters:
        data (TrajectoryCollection) : trajectories
    Returns:
        distances (dict) : dictionary with a pd.Series of distances for each pair in meters

    """
    number_of_animals = len(data.trajectories)
    distances = {}
    for pair in itertools.combinations(range(number_of_animals), 2):
        distance = calculate_distance_between_trajectories(data.trajectories[pair[0]].df, data.trajectories[pair[1]].df)
        distances[pair] = distance
    return distances


def count_number_of_samples_when_close(data: TrajectoryCollection, distances: dict, meeting_distance=100) -> dict:
    """
    Count the number of times a pair of animals got close to each other
    Parameters:
        data (TrajectoryCollection): animal trajectories
        distances (dict): dictionary with series of pairwise distances
        meeting_distance (int): threshold for considering two animals to be close (meters)
    Returns:
        samples_when_close (dict): dictionary of number of samples when pairs of animals were close to each other
    """
    number_of_animals = len(data.trajectories)
    samples_when_close = {}
    for pair in itertools.combinations(range(number_of_animals), 2):
        print(pair)
        distance = distances[pair]  # data frame with distances
        samples_when_close[(data.trajectories[pair[0]].id, data.trajectories[pair[1]].id)] = (distance < meeting_distance).sum()
        print((distance < meeting_distance).sum())
    return samples_when_close


def create_graph(data: TrajectoryCollection):
    logging.info("Create relationship graph")
    distances = calculate_distances_between_pairs_of_trajectories(data)
    samples_when_close = count_number_of_samples_when_close(data, distances)

    relationship_graph = nx.Graph()
    for key in samples_when_close:
        # add edges
        relationship_graph.add_edge(str(key[0]), str(key[1]), weight=samples_when_close[key])
    pos = nx.spring_layout(relationship_graph, seed=111)

    # nodes
    nx.draw_networkx_nodes(relationship_graph, pos)

    # edges
    widths = nx.get_edge_attributes(relationship_graph, 'weight')

    #nx.draw_networkx_edges(
    #    relationship_graph, pos, edgelist=widths.keys(), width=list(widths.values()), alpha=0.3, edge_color='skyblue'
    #)

    # node labels
    #nx.draw_networkx_labels(relationship_graph, pos, font_size=12, font_family="sans-serif")  # todo make it look nicer
    # edge weight labels
    edge_labels = nx.get_edge_attributes(relationship_graph, "weight")
    #nx.draw_networkx_edge_labels(relationship_graph, pos, edge_labels)

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
