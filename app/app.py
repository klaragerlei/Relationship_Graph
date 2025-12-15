from sdk.moveapps_spec import hook_impl
from sdk.moveapps_io import MoveAppsIo
from movingpandas import TrajectoryCollection
import logging

from app.create_relationship_graph import create_graph


class App(object):

    def __init__(self, moveapps_io):
        self.moveapps_io = moveapps_io

    @hook_impl
    def execute(self, data: TrajectoryCollection, config: dict) -> TrajectoryCollection:
        logging.info(f'Welcome to the {config}')

        # Create the relationship graph (your main app logic)
        create_graph(data.copy())

        # Return the original data unchanged for the next app in the workflow
        return data