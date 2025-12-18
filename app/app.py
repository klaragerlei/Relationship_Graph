from sdk.moveapps_spec import hook_impl
from movingpandas import TrajectoryCollection
from typing import Optional
import logging

from app.getGeoDataFrame import get_GDF
from app.create_relationship_graph import create_graph


class App(object):

    def __init__(self, moveapps_io):
        self.moveapps_io = moveapps_io

    @hook_impl
    def execute(self, data: TrajectoryCollection, config: dict) -> Optional[TrajectoryCollection]:
        logging.info(f'Welcome to the {config}')

        # Extract all config parameters with defaults
        meeting_distance = config.get("meeting-distance", 20.0)
        time_unit = config.get("time-unit", "h")
        group_id_column = config.get("group-id-column", "group_id")
        edge_threshold_percentile = config.get("edge-threshold-percentile", 40.0)
        node_spacing = config.get("node-spacing", 2.0)
        min_edge_width = config.get("min-edge-width", 0.5)
        max_edge_width = config.get("max-edge-width", 3.5)
        node_size = config.get("node-size", 700)
        font_size = config.get("font-size", 9)
        figure_width = config.get("figure-width", 14)
        figure_height = config.get("figure-height", 10)
        show_edge_labels = config.get("show-edge-labels", False)
        edge_label_threshold_percentile = config.get("edge-label-threshold-percentile", 75.0)
        label_strategy = config.get("label-strategy", "none")

        # Handle empty string for group_id_column (means no grouping)
        if group_id_column == "":
            group_id_column = None

        # Create relationship graph and save it
        fig = create_graph(
            data,
            meeting_distance=meeting_distance,
            time_step=time_unit,
            edge_threshold_percentile=edge_threshold_percentile,
            node_spacing=node_spacing,
            min_edge_width=min_edge_width,
            max_edge_width=max_edge_width,
            node_size=node_size,
            font_size=font_size,
            figsize=(figure_width, figure_height),
            show_edge_labels=show_edge_labels,
            edge_label_threshold_percentile=edge_label_threshold_percentile,
            label_strategy=label_strategy,
            color_by=group_id_column
        )

        if fig is not None:
            fig.savefig(self.moveapps_io.create_artifacts_file('relationship_graph.png'),
                        bbox_inches='tight', dpi=300)
            logging.info('Saved relationship graph')

        # Filter data by year
        data_gdf = get_GDF(data)
        logging.info(f'Subsetting data for {config["year"]}')

        if config["year"] in data_gdf.index.year:
            result_gdf = data_gdf[data_gdf.index.year == config["year"]]
        else:
            # Return empty TrajectoryCollection instead of None
            logging.warning(f'No data found for year {config["year"]}')
            result_gdf = data_gdf[data_gdf.index.year == config["year"]]  # This will be empty

        # Return filtered data as TrajectoryCollection
        result = TrajectoryCollection(
            result_gdf,
            traj_id_col=data.get_traj_id_col(),
            t=data.to_point_gdf().index.name,
            crs=data.get_crs()
        )
        return result