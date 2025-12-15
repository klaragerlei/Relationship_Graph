from sdk.moveapps_spec import hook_impl
from sdk.moveapps_io import MoveAppsIo
from movingpandas import TrajectoryCollection
import logging

from app.create_relationship_graph import create_graph
from app.getGeoDataFrame import get_GDF


class App(object):

    def __init__(self, moveapps_io):
        self.moveapps_io = moveapps_io

    @hook_impl
    def execute(self, data: TrajectoryCollection, config: dict) -> TrajectoryCollection:
        logging.info(f'Welcome to the {config}')

        # Your app logic
        create_graph(data.copy())

        # Convert to GeoDataFrame and back to satisfy the test structure
        data_gdf = get_GDF(data)
        result = TrajectoryCollection(
            data_gdf,
            traj_id_col=data.get_traj_id_col(),
            t=data.to_point_gdf().index.name,
            crs=data.get_crs()
        )

        return result