# Oversabi - Chudi
# Time the rider-minutes stage with Python loops vs NumPy.

import sys
import timeit

import numpy as np

from src.structure import load_data, to_grid
from src.problems import find_and_fix
from src.routing import distance_matrix, nearest_hub, one_way_minutes
from src.riders import busy_minutes, hub_membership, riders_needed
