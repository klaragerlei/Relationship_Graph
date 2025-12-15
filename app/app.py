from sdk.moveapps_spec import hook_impl
from sdk.moveapps_io import MoveAppsIo
from movingpandas import TrajectoryCollection
from typing import Optional
import logging
import matplotlib.pyplot as plt

from app.getGeoDataFrame import get_GDF
from app.create_relationship_graph import create_graph


class App(object):

    def __init__(self, moveapps_io):
        self.moveapps_io = moveapps_io

    @hook_impl
    def execute(self, data: TrajectoryCollection, config: dict) -> Optional[TrajectoryCollection]:
        logging.info(f'Welcome to the {config}')

        create_graph(data.copy())

        # Original filtering logic
        data_gdf = get_GDF(data)

        logging.info(f'Subsetting data for {config["year"]}')
        if config["year"] in data_gdf.index.year:
            result = data_gdf[data_gdf.index.year == config["year"]]
        else:
            return None  # Test expects None when year not present

        # Convert back to TrajectoryCollection
        result = TrajectoryCollection(
            result,
            traj_id_col=data.get_traj_id_col(),
            t=data.to_point_gdf().index.name,
            crs=data.get_crs()
        )

        return result