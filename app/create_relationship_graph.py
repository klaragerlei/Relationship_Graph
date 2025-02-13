import logging
from movingpandas import TrajectoryCollection
import numpy as np


def create_graph(data: TrajectoryCollection):
    logging.info("Create relationship graphs")

    trajectory_a = data.trajectories[0].df
    trajectory_b = data.trajectories[1].df

    # 1: interpolate and resample
    # round (based on param)
    # remove extras (or avg) - there might be multiple identical values after rounding
    # interpolate values / or fill with none depending on gap size if the data is too sparse
    # should return evenly sampled geoseries

    # 2: match series
    # run (1) for both trajectories that are compared and cut off beginnings and ends, so they fully match
    # 3: calculate distance

    distance = trajectory_a.distance(trajectory_b)

    # parameterize and define 'close' - for how long and what distance should they be to be considered interacting?
    # find distance segments that meet criteria
    # get number of times the animals met and store
    # make graph
