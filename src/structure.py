# Data Structure - Diya
# Load the .npz file and give the flat order stream its structure.

import numpy as np
import json
def load_data(path):
    data = np.load(path)
    output = {name: data[name] for name in data.files}
    return output
Hours_per_day = 14
def to_grid(orders, n_areas):
    assert orders.ndim == 1
    per_day = Hours_per_day * n_areas
    assert orders.size % per_day == 0
    n_days = orders.size // per_day
    grid = orders.reshape((n_days, Hours_per_day, n_areas))
    return grid


    
