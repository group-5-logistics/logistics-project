# Oversabi - Chudi
# Time the rider-minutes stage with Python loops vs NumPy.
# Usage: python3 benchmark.py data/group05_logistics.npz

import sys
import timeit

import numpy as np

from src.structure import load_data, to_grid
from src.problems import find_and_fix
from src.routing import distance_matrix, nearest_hub, one_way_minutes
from src.riders import busy_minutes, hub_membership, riders_needed


def rider_minutes_loops(grid, busy, nearest, n_hubs):
    """Same sum as riders_needed, written with plain Python loops.

    in:  grid (days, 14, 12), busy (14, 12), nearest (12,), number of hubs
    out: nested list [days][14][hubs] of rider-minutes
    """
    n_days, n_hours, n_areas = grid.shape
    out = [[[0.0] * n_hubs for _ in range(n_hours)] for _ in range(n_days)]
    for d in range(n_days):
        for h in range(n_hours):
            for a in range(n_areas):
                out[d][h][nearest[a]] += grid[d, h, a] * busy[h, a]
    return out


def main(path):
    raw = load_data(path)
    params, traffic = raw["params"], raw["traffic_factor_by_hour"]
    n_hubs = len(raw["hubs"])

    grid = to_grid(raw["orders"], raw["areas"].size, traffic.size)
    clean = find_and_fix(grid, raw["area_xy_km"], traffic)
    nearest, d_min = nearest_hub(distance_matrix(clean["area_xy"], raw["hub_xy_km"]))
    busy = busy_minutes(one_way_minutes(d_min, clean["traffic"], params), params["handover_minutes"])
    membership = hub_membership(nearest, hub_count=n_hubs)

    # Same answer both ways (nan days compared as nan)
    loop = np.array(rider_minutes_loops(clean["grid"], busy, nearest, n_hubs))    # (56, 14, 3)
    vec, _ = riders_needed(clean["grid"], busy, membership)                       # (56, 14, 3)
    assert np.allclose(loop, vec, equal_nan=True)
    print("Loops and NumPy give the same rider-minutes")

    n = 20
    t_loop = timeit.timeit(lambda: rider_minutes_loops(clean["grid"], busy, nearest, n_hubs), number=n) / n
    t_np = timeit.timeit(lambda: riders_needed(clean["grid"], busy, membership), number=n) / n
    print(f"Python loops: {t_loop * 1000:.2f} ms per run")
    print(f"NumPy:        {t_np * 1000:.3f} ms per run")
    print(f"Speed-up:     {t_loop / t_np:.0f}x")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "data/group05_logistics.npz")
