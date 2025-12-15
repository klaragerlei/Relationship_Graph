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

        # Create relationship graph and save it
        fig = create_graph(data)
        if fig is not None:
            fig.savefig(self.moveapps_io.create_artifacts_file('relationship_graph.png'),
                        bbox_inches='tight', dpi=300)
            logging.info('Saved relationship graph')

        # Filter data by year
        data_gdf = get_GDF(data)
        logging.info(f'Subsetting data for {config["year"]}')

        if config["year"] in data_gdf.index.year:
            result = data_gdf[data_gdf.index.year == config["year"]]
        else:
            return None

        # Return filtered data as TrajectoryCollection
        result = TrajectoryCollection(
            result,
            traj_id_col=data.get_traj_id_col(),
            t=data.to_point_gdf().index.name,
            crs=data.get_crs()
        )

        return result