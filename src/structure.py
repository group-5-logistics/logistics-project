# Data Structure - Diya
# Load the .npz file and give the flat order stream its structure.

import json

import numpy as np


def load_data(path):
    """Read every array in the .npz file into a dict.

    in:  path to the .npz file
    out: dict of arrays under their npz names, plus first_hour as an int
         and params (dict) read from params_json
    """
    with np.load(path) as data:
        output = {name: data[name] for name in data.files}
    output["first_hour"] = int(output["first_hour"])
    output["params"] = json.loads(str(output["params_json"]))
    return output


def to_grid(orders, n_areas, n_hours):
    """Turn the flat order stream into one block per day.

    in:  orders (N,), number of areas, number of hours in a day
    out: grid (days, 14, 12)
    """
    assert orders.ndim == 1
    per_day = n_hours * n_areas
    assert orders.size % per_day == 0
    n_days = orders.size // per_day
    grid = orders.reshape((n_days, n_hours, n_areas))
    return grid
