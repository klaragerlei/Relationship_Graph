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
from typing import Dict, List, Tuple, Optional


def round_time(dates: pd.Series, unit: str = 'H') -> pd.Series:
    """
    Round series of dates to specified time unit.

    Parameters:
        dates (pd.Series): Series of datetime values
        unit (str): Time unit for rounding ('H', 'min', 'D', etc.)
    Returns:
        pd.Series: Rounded datetime series
    """
    return dates.floor(unit)


def interpolate_points(trajectory: gpd.GeoDataFrame, time_unit: str = 'h') -> pd.DataFrame:
    """
    Creates data frame with time and x and y coordinates and resamples time series data.

    Parameters:
        trajectory (gpd.GeoDataFrame): trajectory data
        time_unit (str): unit of time for resampling
    Returns:
        trajectory_df (pd.DataFrame): resampled trajectory
    """
    trajectory_df = pd.DataFrame({'x': trajectory.geometry.x, 'y': trajectory.geometry.y},
                                 index=trajectory.index)
    trajectory_df = trajectory_df.resample(time_unit).mean()
    return trajectory_df


def remove_duplicate_times(trajectory: pd.DataFrame) -> pd.DataFrame:
    """
    Remove duplicate time indices from trajectory, keeping first occurrence.

    Parameters:
        trajectory (pd.DataFrame): Trajectory data with datetime index
    Returns:
        pd.DataFrame: Trajectory with unique time indices
    """
    return trajectory[~trajectory.index.duplicated(keep='first')]


def align_trajectories(traj_a: pd.DataFrame, traj_b: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Align two trajectories to have matching time indices.

    Parameters:
        traj_a (pd.DataFrame): First trajectory
        traj_b (pd.DataFrame): Second trajectory
    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: Aligned trajectories with matching indices
    """
    return traj_a.align(traj_b, join="outer", axis=0)


def convert_to_geodataframe(df: pd.DataFrame, from_crs: str = 'EPSG:4326',
                            to_crs: str = 'EPSG:3857') -> gpd.GeoDataFrame:
    """
    Convert DataFrame with x,y coordinates to GeoDataFrame and reproject.

    Parameters:
        df (pd.DataFrame): DataFrame with 'x' and 'y' columns
        from_crs (str): Source coordinate reference system
        to_crs (str): Target coordinate reference system
    Returns:
        gpd.GeoDataFrame: Reprojected GeoDataFrame
    """
    gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df['x'], df['y']))
    return gdf.set_crs(from_crs).to_crs(to_crs)


def calculate_distance_between_trajectories(trajectory_a: pd.DataFrame, trajectory_b: pd.DataFrame,
                                            time_step: str = 'h') -> pd.Series:
    """
    Calculates the distance between two trajectories after aligning them.

    Parameters:
        trajectory_a (pd.DataFrame): data frame with trajectory
        trajectory_b (pd.DataFrame): data frame with trajectory
        time_step (str): unit of time used for interpolation ('h' by default)
    Returns:
        distance (pd.Series): distance between trajectory a and b at each time point
    """
    # Round time indices
    trajectory_a.index = round_time(trajectory_a.index, time_step)
    trajectory_b.index = round_time(trajectory_b.index, time_step)

    # Remove duplicates
    times_a_unique = remove_duplicate_times(trajectory_a)
    times_b_unique = remove_duplicate_times(trajectory_b)

    # Interpolate points
    times_a_interpolated = interpolate_points(times_a_unique, time_unit=time_step)
    times_b_interpolated = interpolate_points(times_b_unique, time_unit=time_step)

    # Align trajectories
    a_aligned, b_aligned = align_trajectories(times_a_interpolated, times_b_interpolated)

    # Convert to GeoDataFrame and calculate distance
    a_gdf = convert_to_geodataframe(a_aligned)
    b_gdf = convert_to_geodataframe(b_aligned)
    distance = a_gdf.distance(b_gdf)  # meters

    return distance


def calculate_distances_between_pairs_of_trajectories(data: TrajectoryCollection,
                                                      time_step: str = 'h') -> Dict[Tuple[int, int], pd.Series]:
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


def count_close_encounters(distance_series: pd.Series, meeting_distance: float) -> int:
    """
    Count number of time points where distance is below threshold.

    Parameters:
        distance_series (pd.Series): Series of distances over time
        meeting_distance (float): Distance threshold in meters
    Returns:
        int: Number of close encounters
    """
    return (distance_series < meeting_distance).sum()


def count_number_of_samples_when_close(data: TrajectoryCollection, distances: Dict[Tuple[int, int], pd.Series],
                                       meeting_distance: float = 20) -> Dict[Tuple[str, str], int]:
    """
    Count the number of times a pair of animals got close to each other.

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
        distance = distances[pair]
        id_pair = (data.trajectories[pair[0]].id, data.trajectories[pair[1]].id)
        samples_when_close[id_pair] = count_close_encounters(distance, meeting_distance)
    return samples_when_close


def build_graph_from_samples(samples_when_close: Dict[Tuple[str, str], int]) -> nx.Graph:
    """
    Build NetworkX graph from close encounter samples.

    Parameters:
        samples_when_close (dict): Dictionary mapping ID pairs to encounter counts
    Returns:
        nx.Graph: Graph with nodes and weighted edges
    """
    graph = nx.Graph()
    for key, weight in samples_when_close.items():
        graph.add_edge(str(key[0]), str(key[1]), weight=weight)
    return graph


def get_edge_weights(graph: nx.Graph) -> Dict[Tuple[str, str], float]:
    """
    Extract edge weights from graph.

    Parameters:
        graph (nx.Graph): NetworkX graph
    Returns:
        dict: Dictionary mapping edge tuples to weights
    """
    return nx.get_edge_attributes(graph, 'weight')


def calculate_edge_threshold(weight_values: List[float], percentile: float) -> float:
    """
    Calculate edge weight threshold based on percentile.

    Parameters:
        weight_values (list): List of edge weights
        percentile (float): Percentile value (0-100)
    Returns:
        float: Threshold value
    """
    return np.percentile(weight_values, percentile)


def filter_edges_by_threshold(widths: Dict[Tuple[str, str], float],
                              threshold: float) -> List[Tuple[str, str]]:
    """
    Filter edges to keep only those above threshold.

    Parameters:
        widths (dict): Dictionary of edge weights
        threshold (float): Minimum weight threshold
    Returns:
        list: List of edge tuples that pass threshold
    """
    return [(u, v) for (u, v), w in widths.items() if w >= threshold]


def build_filtered_graph(widths: Dict[Tuple[str, str], float], threshold: float) -> nx.Graph:
    """
    Build new graph containing only edges above threshold.

    Parameters:
        widths (dict): Dictionary of edge weights
        threshold (float): Minimum weight threshold
    Returns:
        nx.Graph: Filtered graph
    """
    filtered_graph = nx.Graph()
    for (u, v), w in widths.items():
        if w >= threshold:
            filtered_graph.add_edge(u, v, weight=w)
    return filtered_graph


def calculate_graph_layout(graph: nx.Graph, node_spacing: float) -> Dict[str, Tuple[float, float]]:
    """
    Calculate spring layout positions for graph nodes.

    Parameters:
        graph (nx.Graph): NetworkX graph
        node_spacing (float): Spacing parameter for layout
    Returns:
        dict: Dictionary mapping node IDs to (x, y) positions
    """
    return nx.spring_layout(graph, seed=111, k=node_spacing, iterations=100, weight='weight')


def normalize_edge_widths(weight_values: List[float], min_edge_width: float,
                          max_edge_width: float) -> List[float]:
    """
    Normalize edge weights to visualization width range.

    Parameters:
        weight_values (list): List of edge weights
        min_edge_width (float): Minimum visualization width
        max_edge_width (float): Maximum visualization width
    Returns:
        list: List of normalized widths
    """
    min_weight = np.percentile(weight_values, 10)
    max_weight = np.percentile(weight_values, 90)

    normalized_widths = [
        min_edge_width + (max_edge_width - min_edge_width) *
        (min(max(w, min_weight), max_weight) - min_weight) / (max_weight - min_weight + 1e-10)
        for w in weight_values
    ]
    return normalized_widths


def filter_widths_by_threshold(normalized_widths: List[float], widths: Dict[Tuple[str, str], float],
                               threshold: float) -> List[float]:
    """
    Filter normalized widths to match filtered edges.

    Parameters:
        normalized_widths (list): List of all normalized widths
        widths (dict): Dictionary of edge weights
        threshold (float): Edge weight threshold
    Returns:
        list: Filtered list of widths
    """
    return [
        normalized_widths[i]
        for i, ((u, v), w) in enumerate(widths.items())
        if w >= threshold
    ]


def extract_node_groups(data: TrajectoryCollection, graph: nx.Graph,
                        color_by: str) -> Dict[str, any]:
    """
    Extract group assignments for nodes from trajectory data.

    Parameters:
        data (TrajectoryCollection): Trajectory data
        graph (nx.Graph): Graph containing nodes
        color_by (str): Column name to use for grouping
    Returns:
        dict: Dictionary mapping node IDs to group values
    """
    node_to_group = {}
    for traj in data.trajectories:
        traj_id = str(traj.id)
        if traj_id in graph.nodes() and color_by in traj.df.columns:
            group_id = traj.df[color_by].unique()[0]
            node_to_group[traj_id] = group_id
    return node_to_group


def create_color_map(unique_groups: List[any]) -> Dict[any, str]:
    """
    Create color mapping for unique groups.

    Parameters:
        unique_groups (list): List of unique group values
    Returns:
        dict: Dictionary mapping group values to hex colors
    """
    cmap = cm.get_cmap('tab10' if len(unique_groups) <= 10 else 'tab20')
    color_map = {
        group: mcolors.rgb2hex(cmap(i / max(len(unique_groups) - 1, 1)))
        for i, group in enumerate(unique_groups)
    }
    return color_map


def assign_node_colors(graph: nx.Graph, node_to_group: Dict[str, any],
                       color_map: Dict[any, str]) -> List[str]:
    """
    Assign colors to nodes based on group membership.

    Parameters:
        graph (nx.Graph): Graph containing nodes
        node_to_group (dict): Mapping of nodes to groups
        color_map (dict): Mapping of groups to colors
    Returns:
        list: List of color strings for each node
    """
    return [color_map.get(node_to_group.get(node), 'lightgrey')
            for node in graph.nodes()]


def get_node_colors(data: TrajectoryCollection, graph: nx.Graph,
                    color_by: Optional[str]) -> Tuple[any, Dict[any, str]]:
    """
    Determine node colors based on group attribute.

    Parameters:
        data (TrajectoryCollection): Trajectory data
        graph (nx.Graph): Graph to color
        color_by (str): Attribute name to color by, or None
    Returns:
        Tuple: (node_colors, color_map) - colors for each node and group color mapping
    """
    node_colors = 'lightgrey'
    color_map = {}

    if not color_by:
        return node_colors, color_map

    node_to_group = extract_node_groups(data, graph, color_by)

    if node_to_group:
        unique_groups = sorted(set(node_to_group.values()))
        color_map = create_color_map(unique_groups)
        node_colors = assign_node_colors(graph, node_to_group, color_map)
        logging.info(f"Colored {len(node_colors)} nodes by '{color_by}' with {len(unique_groups)} unique groups")

    return node_colors, color_map


def draw_graph_edges(ax, graph: nx.Graph, pos: Dict, edges: List[Tuple[str, str]],
                     widths: List[float]) -> None:
    """
    Draw edges on the graph.

    Parameters:
        ax: Matplotlib axis
        graph (nx.Graph): NetworkX graph
        pos (dict): Node positions
        edges (list): List of edges to draw
        widths (list): List of edge widths
    """
    nx.draw_networkx_edges(
        graph, pos, edgelist=edges, width=widths,
        alpha=0.3, edge_color='skyblue', ax=ax
    )


def draw_graph_nodes(ax, graph: nx.Graph, pos: Dict, node_colors: any,
                     node_size: int) -> None:
    """
    Draw nodes on the graph.

    Parameters:
        ax: Matplotlib axis
        graph (nx.Graph): NetworkX graph
        pos (dict): Node positions
        node_colors: Color or list of colors for nodes
        node_size (int): Size of nodes
    """
    nx.draw_networkx_nodes(
        graph, pos, node_color=node_colors, node_size=node_size,
        alpha=0.9, edgecolors='darkgrey', linewidths=2, ax=ax
    )


def calculate_label_offset_position(x: float, y: float, radius: float) -> Tuple[float, float]:
    """
    Calculate offset position for node label.

    Parameters:
        x (float): Node x position
        y (float): Node y position
        radius (float): Base offset radius
    Returns:
        Tuple[float, float]: Label (x, y) position
    """
    angle = np.arctan2(y, x) + np.random.uniform(-0.3, 0.3)
    offset_dist = radius * 2.5
    label_x = x + offset_dist * np.cos(angle)
    label_y = y + offset_dist * np.sin(angle)
    return label_x, label_y


def draw_label_with_line(ax, node: str, x: float, y: float, label_x: float,
                         label_y: float, font_size: int) -> None:
    """
    Draw node label with connecting line.

    Parameters:
        ax: Matplotlib axis
        node (str): Node ID
        x (float): Node x position
        y (float): Node y position
        label_x (float): Label x position
        label_y (float): Label y position
        font_size (int): Font size for label
    """
    ax.plot([x, label_x], [y, label_y], 'gray', linewidth=0.5, alpha=0.5, zorder=1)
    ax.text(label_x, label_y, node, fontsize=font_size - 1,
            fontweight='bold', ha='center', va='center',
            bbox=dict(boxstyle='round,pad=0.2', facecolor='white',
                      edgecolor='lightgray', alpha=0.9), zorder=2)


def _draw_labels_offset(ax, graph: nx.Graph, pos: Dict, font_size: int, node_size: int) -> None:
    """
    Draw labels offset from nodes with connecting lines.

    Parameters:
        ax: Matplotlib axis
        graph (nx.Graph): NetworkX graph
        pos (dict): Node positions
        font_size (int): Font size for labels
        node_size (int): Size of nodes
    """
    radius = np.sqrt(node_size) / 50.0

    for node, (x, y) in pos.items():
        label_x, label_y = calculate_label_offset_position(x, y, radius)
        draw_label_with_line(ax, node, x, y, label_x, label_y, font_size)


def _draw_labels_minimal(ax, graph: nx.Graph, pos: Dict, font_size: int) -> None:
    """
    Only show labels for nodes with low degree (peripheral nodes).

    Parameters:
        ax: Matplotlib axis
        graph (nx.Graph): NetworkX graph
        pos (dict): Node positions
        font_size (int): Font size for labels
    """
    degrees = dict(graph.degree())
    median_degree = np.median(list(degrees.values()))

    for node, (x, y) in pos.items():
        if degrees[node] <= median_degree:
            ax.text(x, y, node, fontsize=font_size, fontweight='bold',
                    ha='center', va='center',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                              edgecolor='none', alpha=0.8))


def draw_labels_by_strategy(ax, graph: nx.Graph, pos: Dict, font_size: int,
                            node_size: int, label_strategy: str) -> None:
    """
    Draw labels according to specified strategy.

    Parameters:
        ax: Matplotlib axis
        graph (nx.Graph): NetworkX graph
        pos (dict): Node positions
        font_size (int): Font size for labels
        node_size (int): Size of nodes
        label_strategy (str): 'offset', 'minimal', or 'none'
    """
    if label_strategy == 'offset':
        _draw_labels_offset(ax, graph, pos, font_size, node_size)
    elif label_strategy == 'minimal':
        _draw_labels_minimal(ax, graph, pos, font_size)
    elif label_strategy == 'none':
        pass
    else:
        logging.warning(f"Unknown label strategy: {label_strategy}, defaulting to 'offset'")
        _draw_labels_offset(ax, graph, pos, font_size, node_size)


def add_graph_legend(ax, color_map: Dict, color_by: str, font_size: int) -> None:
    """
    Add legend to graph showing group colors.

    Parameters:
        ax: Matplotlib axis
        color_map (dict): Mapping of groups to colors
        color_by (str): Name of grouping attribute
        font_size (int): Font size for legend
    """
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=color, edgecolor='darkgrey', label=str(group))
                       for group, color in sorted(color_map.items())]
    ax.legend(handles=legend_elements,
              title=color_by.replace('_', ' ').title(),
              loc='upper left', bbox_to_anchor=(1.02, 1),
              frameon=True, fancybox=True, shadow=True,
              prop={'size': font_size})


def get_strong_edge_labels(graph: nx.Graph, weight_values: List[float],
                           threshold_percentile: float) -> Dict[Tuple[str, str], str]:
    """
    Get edge labels for edges above threshold.

    Parameters:
        graph (nx.Graph): NetworkX graph
        weight_values (list): List of all edge weights
        threshold_percentile (float): Percentile threshold
    Returns:
        dict: Dictionary of edge labels
    """
    edge_labels = nx.get_edge_attributes(graph, "weight")
    strong_threshold = np.percentile(weight_values, threshold_percentile)
    return {k: f"{v:.0f}" for k, v in edge_labels.items() if v >= strong_threshold}


def draw_edge_labels(ax, graph: nx.Graph, pos: Dict, weight_values: List[float],
                     edge_label_threshold_percentile: float, font_size: int) -> None:
    """
    Draw edge weight labels on graph.

    Parameters:
        ax: Matplotlib axis
        graph (nx.Graph): NetworkX graph
        pos (dict): Node positions
        weight_values (list): List of edge weights
        edge_label_threshold_percentile (float): Percentile threshold for showing labels
        font_size (int): Font size for labels
    """
    strong_edge_labels = get_strong_edge_labels(graph, weight_values, edge_label_threshold_percentile)
    nx.draw_networkx_edge_labels(graph, pos, strong_edge_labels,
                                 font_size=font_size - 2, ax=ax)


def set_graph_title(ax, color_by: Optional[str], color_map: Dict) -> None:
    """
    Set title for graph visualization.

    Parameters:
        ax: Matplotlib axis
        color_by (str): Attribute used for coloring, or None
        color_map (dict): Color mapping dictionary
    """
    title = "Trajectory Relationship Graph (Filtered)"
    if color_by and color_map:
        title += f" - Colored by {color_by}"
    ax.set_title(title, fontsize=14, fontweight='bold')


def log_graph_statistics(relationship_graph: nx.Graph, filtered_graph: nx.Graph,
                         filtered_edges: List, removed_edges: List) -> None:
    """
    Log statistics about graph filtering.

    Parameters:
        relationship_graph (nx.Graph): Original graph
        filtered_graph (nx.Graph): Filtered graph
        filtered_edges (list): List of edges that passed filter
        removed_edges (list): List of edges that were removed
    """
    total_nodes = len(relationship_graph.nodes())
    total_edges = len(relationship_graph.edges())
    logging.info(f"Original graph - Nodes: {total_nodes}, Edges: {total_edges}")

    nodes_in_filtered = set(filtered_graph.nodes())
    removed_nodes = set(relationship_graph.nodes()) - nodes_in_filtered

    logging.info(f"Filtered graph - Nodes: {len(nodes_in_filtered)}, Edges: {len(filtered_edges)}")
    logging.info(f"Removed - Nodes: {len(removed_nodes)}, Edges: {len(removed_edges)}")

    if removed_nodes:
        logging.info(f"Removed nodes: {sorted(removed_nodes)}")

    if removed_edges:
        logging.info(f"Removed edges (showing first 10):")
        for u, v, w in sorted(removed_edges, key=lambda x: x[2])[:10]:
            logging.info(f"  {u} -- {v}: weight={w:.2f}")
        if len(removed_edges) > 10:
            logging.info(f"  ... and {len(removed_edges) - 10} more")


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
        figsize: Tuple[int, int] = (14, 10),
        show_edge_labels: bool = False,
        edge_label_threshold_percentile: float = 75,
        label_strategy: str = 'none',
        color_by: Optional[str] = 'group_id'
) -> plt.Figure:
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
        Strategy for handling labels: 'offset', 'minimal', or 'none'
    color_by : str
        Attribute name to color nodes by (e.g., 'group_id'). Set to None for grey nodes.

    Returns:
    --------
    plt.Figure
        Matplotlib figure containing the relationship graph
    """
    logging.info(f"Create relationship graph with meeting_distance={meeting_distance}m, time_step={time_step}")

    # Calculate distances and build graph
    distances = calculate_distances_between_pairs_of_trajectories(data, time_step=time_step)
    samples_when_close = count_number_of_samples_when_close(data, distances, meeting_distance=meeting_distance)
    relationship_graph = build_graph_from_samples(samples_when_close)

    # Get edge weights and check for empty graph
    widths = get_edge_weights(relationship_graph)
    weight_values = list(widths.values())

    if not weight_values:
        logging.warning("No edges found in graph")
        return None

    # Calculate and apply edge threshold
    edge_threshold = calculate_edge_threshold(weight_values, edge_threshold_percentile)
    logging.info(f"Edge weight threshold (p{edge_threshold_percentile}): {edge_threshold:.2f}")
    logging.info(f"Edge weight range: [{min(weight_values):.2f}, {max(weight_values):.2f}]")

    filtered_edges = filter_edges_by_threshold(widths, edge_threshold)
    removed_edges = [(u, v, w) for (u, v), w in widths.items() if w < edge_threshold]
    filtered_graph = build_filtered_graph(widths, edge_threshold)

    # Log statistics
    log_graph_statistics(relationship_graph, filtered_graph, filtered_edges, removed_edges)

    # Use filtered graph for layout if it has nodes, otherwise use original
    graph_for_layout = filtered_graph if len(filtered_graph.nodes()) > 0 else relationship_graph

    # Calculate layout and edge widths
    pos = calculate_graph_layout(graph_for_layout, node_spacing)
    normalized_widths = normalize_edge_widths(weight_values, min_edge_width, max_edge_width)
    filtered_widths = filter_widths_by_threshold(normalized_widths, widths, edge_threshold)

    # Get node colors
    node_colors, color_map = get_node_colors(data, graph_for_layout, color_by)

    # Create figure and draw graph
    fig, ax = plt.subplots(figsize=figsize)
    draw_graph_edges(ax, graph_for_layout, pos, filtered_edges, filtered_widths)
    draw_graph_nodes(ax, graph_for_layout, pos, node_colors, node_size)

    # Draw labels
    draw_labels_by_strategy(ax, graph_for_layout, pos, font_size, node_size, label_strategy)

    # Add legend if we have colored groups
    if color_by and color_map:
        add_graph_legend(ax, color_map, color_by, font_size)

    # Optionally add edge labels
    if show_edge_labels:
        draw_edge_labels(ax, graph_for_layout, pos, weight_values,
                         edge_label_threshold_percentile, font_size)

    # Set title and finalize
    set_graph_title(ax, color_by, color_map)
    ax.axis('off')
    plt.tight_layout()

    return fig