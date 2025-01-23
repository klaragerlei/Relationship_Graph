import logging
from movingpandas import TrajectoryCollection
import numpy as np


def create_graph(data: TrajectoryCollection):
    logging.info("Create relationship graphs")
    all_z_coord_missing = data.trajectories[0].df.geometry.z.isnull().all()
    if all_z_coord_missing:
        # force2d oly works for geoseries..?
        trajectory_a = data.trajectories[0].df.force_2d()
        trajectory_b = data.trajectories[1].df.force_2d()
        distance = trajectory_a.distance(trajectory_b)
        print(distance)


        # check if there is a 3rd dimension
        # for each pair:
        # get geoseries for 2 trajectories
        # use force_2d()
        # use a.distance(b) to get distances between points
        # parameterize and define close - for how long and what distance?
        # find distance segments that meet criteria
        # get number of times the animals met and store
        # make graph
