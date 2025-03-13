import logging
from movingpandas import TrajectoryCollection
import geopandas as gpd
import numpy as np

from datetime import datetime
from datetime import timedelta

import pandas as pd


def round_time(dates: datetime, unit='H'):
    return dates.floor(unit)


def interpolate_points(trajectory: gpd.GeoDataFrame, unit='H'):
    # make new df with time, x and y coord
    trajectory_df = pd.DataFrame({'x': trajectory.geometry.x, 'y': trajectory.geometry.y}, index=trajectory.index)
    # use pandas interpolate
    trajectory_df = trajectory_df.resample(unit).mean()   # this has nans in gaps for now
    return trajectory_df


def create_graph(data: TrajectoryCollection):
    logging.info("Create relationship graphs")

    trajectory_a = data.trajectories[0].df
    trajectory_b = data.trajectories[1].df

    # 1: interpolate and resample
    # round (based on param) 'T', 'M' or 'H'
    times_a_rounded_time = round_time(trajectory_a.index, 'H')
    times_b_rounded_time = round_time(trajectory_b.index, 'H')

    trajectory_a.index = times_a_rounded_time
    trajectory_b.index = times_b_rounded_time

    # remove extras (or avg) - there might be multiple identical values after rounding
    times_a_unique = trajectory_a[~trajectory_a.index.duplicated(keep='first')]
    times_b_unique = trajectory_b[~trajectory_b.index.duplicated(keep='first')]

    # interpolate values / or fill with none depending on gap size if the data is too sparse
    times_a_interpolated = interpolate_points(times_a_unique)
    times_b_interpolated = interpolate_points(times_b_unique)


    # should return evenly sampled geoseries

    # 2: match series
    # run (1) for both trajectories that are compared and cut off beginnings and ends, so they fully match
    # 3: calculate distance

    distance = trajectory_a.distance(trajectory_b)

    # parameterize and define 'close' - for how long and what distance should they be to be considered interacting?
    # find distance segments that meet criteria
    # get number of times the animals met and store
    # make graph
