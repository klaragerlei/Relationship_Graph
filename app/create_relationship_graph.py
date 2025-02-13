import logging
from movingpandas import TrajectoryCollection
import numpy as np


def create_graph(data: TrajectoryCollection):
    logging.info("Create relationship graphs")

    trajectory_a = data.trajectories[0].df
    trajectory_b = data.trajectories[1].df

    # 1: interpolate and resample
    # round
    # remove extras (or avg)
    # interpolate / fill with none depending on gap size

    # 2: match series

    # 3: if it's within params, calculate distance
    # create now df with times they were both sampled with distance

    # calculate distance
    distance = trajectory_a.distance(trajectory_b)

    # parameterize and define close - for how long and what distance?
    # find distance segments that meet criteria
    # get number of times the animals met and store
    # make graph
