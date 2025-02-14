import logging
from movingpandas import TrajectoryCollection
import numpy as np

from datetime import datetime
from datetime import timedelta
def round_time(dt: datetime, unit=timedelta(seconds=1)):
    seconds = (dt - datetime.min).total_seconds()  #todo fix types here
    unit_seconds = unit.total_seconds()
    half_over = seconds + unit_seconds / 2
    rounded_seconds = half_over - half_over % unit_seconds
    return datetime.min + timedelta(seconds=rounded_seconds)



def create_graph(data: TrajectoryCollection):
    logging.info("Create relationship graphs")

    trajectory_a = data.trajectories[0].df
    trajectory_b = data.trajectories[1].df
    t1 = trajectory_b.index.values[0]
    t_rounded = round_time(t1, unit=timedelta(hours=5))
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
