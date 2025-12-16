import logging
from movingpandas import TrajectoryCollection
import geopandas as gpd
import itertools
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import networkx as nx
import pandas as pd
import numpy as np


def round_time(dates: pd.Series, unit='H'):
    """
    Round series of dates
    """
    return dates.floor(unit)


def interpolate_points(trajectory: gpd.GeoDataFrame, time_unit='h') -> pd.DataFrame:
    """
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
    trajectory_df = trajectory_df.resample(time_unit).mean()  # this has nans in gaps
    return trajectory_df


def calculate_distance_between_trajectories(trajectory_a: pd.DataFrame, trajectory_b: pd.DataFrame,
                                            time_step='h') -> pd.Series:
    """
    Calculates the distance between two trajectories after aligning them

    Parameters:
        trajectory_a (pd.DataFrame): data frame with trajectory
        trajectory_b (pd.DataFrame): data frame with trajectory
        time_step (str): unit of time used for interpolation ('h' by default)
    Returns:
        distance (pd.Series): distance between trajectory a and b at each time point
    """
    # round (based on param)
    times_a_rounded_time = round_time(trajectory_a.index, time_step)
    times_b_rounded_time = round_time(trajectory_b.index, time_step)

    trajectory_a.index = times_a_rounded_time
    trajectory_b.index = times_b_rounded_time

    # remove duplicates - there might be multiple identical values after rounding
    times_a_unique = trajectory_a[~trajectory_a.index.duplicated(keep='first')]
    times_b_unique = trajectory_b[~trajectory_b.index.duplicated(keep='first')]

    # resample and fill gaps with missing values
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


def calculate_distances_between_pairs_of_trajectories(data: TrajectoryCollection, time_step='h') -> dict:
    """
    Calculates the distance between pairs of trajectories for all trajectories in the data.

    Parameters:
        data (TrajectoryCollection): trajectories
        time_step (str): unit of time used for interpolation ('h' by default)
    Returns:
        distances (dict): dictionary with a pd.Series of distances for each pair in meters

    """
    number_of_animals = len(data.trajectories)
    distances = {}
    for pair in itertools.combinations(range(number_of_animals), 2):
        distance = calculate_distance_between_trajectories(
            data.trajectories[pair[0]].df,
            data.trajectories[pair[1]].df,
            time_step=time_step
        )
        distances[pair] = distance
    return distances


def count_number_of_samples_when_close(data: TrajectoryCollection, distances: dict, meeting_distance=20) -> dict:
    """
    Count the number of times a pair of animals got close to each other
    Parameters:
        data (TrajectoryCollection): animal trajectories
        distances (dict): dictionary with series of pairwise distances
        meeting_distance (float): threshold for considering two animals to be close (meters)
    Returns:
        samples_when_close (dict): dictionary of number of samples when pairs of animals were close to each other
    """
    number_of_animals = len(data.trajectories)
    samples_when_close = {}
    for pair in itertools.combinations(range(number_of_animals), 2):
        print(pair)
        distance = distances[pair]  # data frame with distances
        samples_when_close[(data.trajectories[pair[0]].id, data.trajectories[pair[1]].id)] = (
                distance < meeting_distance).sum()
        print((distance < meeting_distance).sum())
    return samples_when_close


def create_graph(
        data: TrajectoryCollection,
        meeting_distance: float = 20.0,
        time_step: str = 'h',
        edge_threshold_percentile: float = 40,
        node_spacing: float = 2.0,
        min_edge_width: float = 0.5,
        max_edge_width: float = 3.5,
        node_size: int = 700,
        font_size: int = 9,
        figsize: tuple = (14, 10),
        show_edge_labels: bool = False,
        edge_label_threshold_percentile: float = 75,
        label_strategy: str = 'none',  # 'offset', 'minimal', or 'none'
        color_by: str = 'group_id'  # attribute to color nodes by
):
    """
    Create a relationship graph from trajectory data.

    Parameters:
    -----------
    data : TrajectoryCollection
        The trajectory data to analyze
    meeting_distance : float
        Distance threshold in meters for considering two animals to be close (default: 20.0)
    time_step : str
        Time unit for resampling trajectory data (default: 'h' for hours).
        Common values: 'min' (minutes), 'h' (hours), 'D' (days)
        Uses pandas datetime frequency strings.
    edge_threshold_percentile : float
        Percentile threshold for filtering weak edges (0-100). Higher = fewer edges.
    node_spacing : float
        Spacing between nodes in layout (k parameter). Higher = more spread out.
    min_edge_width : float
        Minimum width for edges in visualization.
    max_edge_width : float
        Maximum width for edges in visualization.
    node_size : int
        Size of nodes in the graph.
    font_size : int
        Font size for node labels.
    figsize : tuple
        Figure size (width, height) in inches.
    show_edge_labels : bool
        Whether to show edge weight labels.
    edge_label_threshold_percentile : float
        Only show labels for edges above this percentile (if show_edge_labels=True).
    label_strategy : str
        Strategy for handling labels:
        - 'offset': Offset labels from nodes with lines
        - 'minimal': Only show labels for peripheral nodes (reduces clutter)
        - 'none': Don't show labels at all
    color_by : str
        Attribute name to color nodes by (e.g., 'group_id'). Set to None for grey nodes.
    """
    logging.info(f"Create relationship graph with meeting_distance={meeting_distance}m, time_step={time_step}")

    distances = calculate_distances_between_pairs_of_trajectories(data, time_step=time_step)
    samples_when_close = count_number_of_samples_when_close(data, distances, meeting_distance=meeting_distance)

    relationship_graph = nx.Graph()
    for key in samples_when_close:
        relationship_graph.add_edge(str(key[0]), str(key[1]), weight=samples_when_close[key])

    total_nodes = len(relationship_graph.nodes())
    total_edges = len(relationship_graph.edges())

    logging.info(f"Original graph - Nodes: {total_nodes}, Edges: {total_edges}")

    # Get edge weights
    widths = nx.get_edge_attributes(relationship_graph, 'weight')
    weight_values = list(widths.values())

    if not weight_values:
        logging.warning("No edges found in graph")
        return relationship_graph, {}

    # Calculate edge threshold
    edge_threshold = np.percentile(weight_values, edge_threshold_percentile)
    logging.info(f"Edge weight threshold (p{edge_threshold_percentile}): {edge_threshold:.2f}")
    logging.info(f"Edge weight range: [{min(weight_values):.2f}, {max(weight_values):.2f}]")

    # Filter edges and track removed nodes
    filtered_edges = [(u, v) for (u, v), w in widths.items() if w >= edge_threshold]
    removed_edges = [(u, v, w) for (u, v), w in widths.items() if w < edge_threshold]

    # Create filtered graph for layout
    filtered_graph = nx.Graph()
    for (u, v), w in widths.items():
        if w >= edge_threshold:
            filtered_graph.add_edge(u, v, weight=w)

    # Find nodes that became isolated after filtering
    nodes_in_filtered_graph = set(filtered_graph.nodes())
    removed_nodes = set(relationship_graph.nodes()) - nodes_in_filtered_graph

    logging.info(f"Filtered graph - Nodes: {len(nodes_in_filtered_graph)}, Edges: {len(filtered_edges)}")
    logging.info(f"Removed - Nodes: {len(removed_nodes)}, Edges: {len(removed_edges)}")

    if removed_nodes:
        logging.info(f"Removed nodes: {sorted(removed_nodes)}")

    if removed_edges:
        logging.info(f"Removed edges (showing first 10):")
        for u, v, w in sorted(removed_edges, key=lambda x: x[2])[:10]:
            logging.info(f"  {u} -- {v}: weight={w:.2f}")
        if len(removed_edges) > 10:
            logging.info(f"  ... and {len(removed_edges) - 10} more")

    # Use filtered graph for layout if it has nodes, otherwise use original
    graph_for_layout = filtered_graph if len(filtered_graph.nodes()) > 0 else relationship_graph

    # Calculate layout
    pos = nx.spring_layout(
        graph_for_layout,
        seed=111,
        k=node_spacing,
        iterations=100,
        weight='weight'
    )

    # Normalize edge widths
    min_weight = np.percentile(weight_values, 10)
    max_weight = np.percentile(weight_values, 90)

    normalized_widths = [
        min_edge_width + (max_edge_width - min_edge_width) *
        (min(max(w, min_weight), max_weight) - min_weight) / (max_weight - min_weight + 1e-10)
        for w in weight_values
    ]

    filtered_widths = [
        normalized_widths[i]
        for i, ((u, v), w) in enumerate(widths.items())
        if w >= edge_threshold
    ]

    # Get node colors based on group_id attribute
    node_colors = 'lightgrey'
    color_map = {}
    if color_by:
        node_to_group = {}
        for traj in data.trajectories:
            traj_id = str(traj.id)
            if traj_id in graph_for_layout.nodes() and color_by in traj.df.columns:
                # Get unique group_id value for this trajectory
                group_id = traj.df[color_by].unique()[0]
                node_to_group[traj_id] = group_id

        if node_to_group:
            # Get all unique groups and assign colors
            unique_groups = sorted(set(node_to_group.values()))
            cmap = cm.get_cmap('tab10' if len(unique_groups) <= 10 else 'tab20')
            color_map = {group: mcolors.rgb2hex(cmap(i / max(len(unique_groups) - 1, 1)))
                         for i, group in enumerate(unique_groups)}
            node_colors = [color_map.get(node_to_group.get(node), 'lightgrey')
                           for node in graph_for_layout.nodes()]
            logging.info(f"Colored {len(node_colors)} nodes by '{color_by}' with {len(unique_groups)} unique groups")

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Draw edges
    nx.draw_networkx_edges(
        graph_for_layout,
        pos,
        edgelist=filtered_edges,
        width=filtered_widths,
        alpha=0.3,
        edge_color='skyblue',
        ax=ax
    )

    # Draw nodes
    nx.draw_networkx_nodes(
        graph_for_layout,
        pos,
        node_color=node_colors,
        node_size=node_size,
        alpha=0.9,
        edgecolors='darkgrey',
        linewidths=2,
        ax=ax
    )

    # Handle labels based on strategy
    if label_strategy == 'offset':
        _draw_labels_offset(ax, graph_for_layout, pos, font_size, node_size)
    elif label_strategy == 'minimal':
        _draw_labels_minimal(ax, graph_for_layout, pos, font_size)
    elif label_strategy == 'none':
        pass  # No labels
    else:
        logging.warning(f"Unknown label strategy: {label_strategy}, defaulting to 'offset'")
        _draw_labels_offset(ax, graph_for_layout, pos, font_size, node_size)

    # Add legend if we have colored groups
    if color_by and color_map:
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor=color, edgecolor='darkgrey', label=str(group))
                           for group, color in sorted(color_map.items())]
        ax.legend(handles=legend_elements,
                  title=color_by.replace('_', ' ').title(),
                  loc='upper left',
                  bbox_to_anchor=(1.02, 1),
                  frameon=True,
                  fancybox=True,
                  shadow=True,
                  prop={'size': font_size})

    # Optionally add edge labels for strong connections
    if show_edge_labels:
        edge_labels = nx.get_edge_attributes(graph_for_layout, "weight")
        strong_threshold = np.percentile(weight_values, edge_label_threshold_percentile)
        strong_edge_labels = {k: f"{v:.0f}" for k, v in edge_labels.items() if v >= strong_threshold}
        nx.draw_networkx_edge_labels(
            graph_for_layout,
            pos,
            strong_edge_labels,
            font_size=font_size - 2,
            ax=ax
        )

    title = "Trajectory Relationship Graph (Filtered)"
    if color_by and color_map:
        title += f" - Colored by {color_by}"
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.axis('off')
    plt.tight_layout()
    return fig


def _draw_labels_offset(ax, graph, pos, font_size, node_size):
    """Draw labels offset from nodes with connecting lines"""
    # Calculate a radius based on node size
    radius = np.sqrt(node_size) / 50.0

    for node, (x, y) in pos.items():
        # Offset based on position in layout (spread labels around nodes)
        angle = np.arctan2(y, x) + np.random.uniform(-0.3, 0.3)

        # Offset distance
        offset_dist = radius * 2.5
        label_x = x + offset_dist * np.cos(angle)
        label_y = y + offset_dist * np.sin(angle)

        # Draw connecting line
        ax.plot([x, label_x], [y, label_y], 'gray', linewidth=0.5, alpha=0.5, zorder=1)

        # Draw label
        ax.text(label_x, label_y, node, fontsize=font_size - 1,
                fontweight='bold', ha='center', va='center',
                bbox=dict(boxstyle='round,pad=0.2',
                          facecolor='white',
                          edgecolor='lightgray',
                          alpha=0.9),
                zorder=2)


def _draw_labels_minimal(ax, graph, pos, font_size):
    """Only show labels for nodes with low degree (peripheral nodes)"""

    degrees = dict(graph.degree())
    median_degree = np.median(list(degrees.values()))

    for node, (x, y) in pos.items():
        # Only label nodes with degree less than median (less crowded)
        if degrees[node] <= median_degree:
            ax.text(x, y, node, fontsize=font_size,
                    fontweight='bold', ha='center', va='center',
                    bbox=dict(boxstyle='round,pad=0.3',
                              facecolor='white',
                              edgecolor='none',
                              alpha=0.8))