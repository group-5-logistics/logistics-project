# Tests - Chudi
# Check the pipeline on changed copies of the data, made in a temporary folder.
# Usage: python3 test_pipeline.py data/group05_logistics.npz

import os
import sys
import tempfile

import numpy as np

from pipeline import run_pipeline
from src.structure import to_grid


def save_copy(arrays, folder, name, orders, area_xy, traffic):
    """Write a copy of the data with new orders, area positions and traffic.

    in:  arrays from the real file, folder, file name, orders (N,), area_xy (12, 2), traffic (14,)
    out: path to the new .npz file
    """
    path = os.path.join(folder, name)
    np.savez(path, **dict(arrays, orders=orders, area_xy_km=area_xy, traffic_factor_by_hour=traffic))
    return path


def main(real_path):
    with np.load(real_path) as data:
        arrays = {name: data[name] for name in data.files}
    areas, traffic = arrays["areas"], arrays["traffic_factor_by_hour"]
    grid = to_grid(arrays["orders"], areas.size, traffic.size)          # (56, 14, 12)

    # The real file has day 6 copied, Maryland in metres and 12:00 missing.
    # Undo all three to get data with no problems.
    no_copy = np.delete(grid, 6, axis=0)                                 # (55, 14, 12)
    maryland = np.flatnonzero(areas == "Maryland")
    xy_km = arrays["area_xy_km"].copy()
    xy_km[maryland] = xy_km[maryland] / 1000
    full_traffic = traffic.copy()
    full_traffic[4] = (traffic[3] + traffic[5]) / 2

    with tempfile.TemporaryDirectory() as folder:
        figures = os.path.join(folder, "figures")

        # (a) the same three problems in new places
        moved_grid = np.insert(no_copy, 21, no_copy[20], axis=0)         # (56, 14, 12) day 21 copies day 20
        ajah = np.flatnonzero(areas == "Ajah")
        moved_xy = xy_km.copy()
        moved_xy[ajah] = moved_xy[ajah] * 1000
        last_hour = traffic.size - 1
        moved_traffic = full_traffic.copy()
        moved_traffic[last_hour] = np.nan
        path = save_copy(arrays, folder, "moved.npz", moved_grid.ravel(), moved_xy, moved_traffic)
        clean = run_pipeline(path, figures, verbose=False)["clean"]

        assert clean["duplicate_days"].tolist() == [21]
        assert np.isnan(clean["grid"][21]).all()
        print("PASS (a) copied day 21 is found and set to nan")

        assert clean["metre_areas"].tolist() == ajah.tolist()
        assert np.allclose(clean["area_xy"], xy_km)
        print("PASS (a) Ajah in metres is found and converted back to km")

        assert clean["missing_hours"].tolist() == [last_hour]
        assert clean["traffic"][last_hour] == full_traffic[last_hour - 1]
        print("PASS (a) missing traffic on the last hour is found and filled")

        # (b) a clean copy with no problems
        path = save_copy(arrays, folder, "clean.npz", no_copy.ravel(), xy_km, full_traffic)
        results = run_pipeline(path, figures, verbose=False)
        clean = results["clean"]
        assert clean["duplicate_days"].size == 0
        assert clean["metre_areas"].size == 0
        assert clean["missing_hours"].size == 0
        assert results["p95_riders"].shape == arrays["hubs"].shape
        print("PASS (b) clean copy: no problems found and the run finishes")

        # (c) the real file
        results = run_pipeline(real_path, figures, verbose=False)
        assert results["p95_riders"].tolist() == [8, 24, 9]
        print("PASS (c) real file gives riders Ikeja 8, Yaba 24, Lekki 9")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "data/group05_logistics.npz")
