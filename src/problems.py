'''
Data Problems - Victor

Finds and fixes the three data problems: a copied day (set to nan),
an area in metres (divided by 1000), and a missing traffic factor
(filled from neighbouring hours).
'''

import numpy as np


# ---------- Problem 1: a day copied over the next day ----------
def find_duplicate_days(grid):
    """Find days that are an exact copy of the day before.

    in:  grid (days, 14, 12)
    out: indices (k,) of the copied days (k can be 0)
    """
    same = (grid[1:] == grid[:-1]).all(axis=(1, 2))   # (days-1,) True where a day equals the day before
    return np.where(same)[0] + 1                      # +1 because we compared from day 1


def fix_duplicate_days(grid):
    """Set every copied day to nan (a copy is not real data).

    in:  grid (days, 14, 12)
    out: fixed grid (days, 14, 12) as float, nan on the copied days
         bad_days (k,) the day numbers that were copies
    """
    bad_days = find_duplicate_days(grid)
    fixed = grid.astype(np.float64)                   # whole numbers cannot hold nan, so use decimals
    fixed[bad_days] = np.nan                          # the whole day becomes nan, no loop needed
    assert fixed.shape == grid.shape
    return fixed, bad_days


# ---------- Problem 2: an area position in metres ----------
def find_metre_areas(area_xy):
    """Find areas whose position looks like metres instead of km.

    in:  area_xy (12, 2)
    out: mask (12,) True where the area is in metres
    """
    size = np.abs(area_xy).max(axis=1)                # (12,) biggest coordinate of each area
    return size > 20 * np.median(size)                # far bigger than a typical area


def fix_metre_areas(area_xy):
    """Divide metre positions by 1000 to get km.

    in:  area_xy (12, 2)
    out: fixed area_xy (12, 2) in km
         bad_areas (k,) the area numbers that were in metres
    """
    fixed = area_xy.astype(np.float64).copy()
    mask = find_metre_areas(fixed)
    fixed[mask] = fixed[mask] / 1000.0
    assert fixed.shape == area_xy.shape
    return fixed, np.where(mask)[0]


# ---------- Problem 3: a missing traffic factor ----------
def find_missing_traffic(traffic):
    """Find hours whose traffic factor is nan.

    in:  traffic (14,)
    out: indices (k,) of the missing hours
    """
    return np.where(np.isnan(traffic))[0]


def fix_missing_traffic(traffic):
    """Fill nan from the neighbouring hours (a straight line between them).

    in:  traffic (14,)
    out: fixed traffic (14,) with no nan
         bad_hours (k,) the hour positions that were missing
    """
    bad_hours = find_missing_traffic(traffic)
    fixed = traffic.astype(np.float64).copy()
    hours = np.arange(fixed.size)
    ok = ~np.isnan(fixed)
    fixed[~ok] = np.interp(hours[~ok], hours[ok], fixed[ok])
    assert not np.isnan(fixed).any()
    return fixed, bad_hours


# ---------- All three together ----------
def find_and_fix(grid, area_xy, traffic):
    """Run all three fixes and keep the evidence for the report.

    in:  grid (days, 14, 12), area_xy (12, 2), traffic (14,)
    out: dict with the cleaned arrays and where each problem was found
    """
    grid_fixed, bad_days = fix_duplicate_days(grid)
    xy_fixed, bad_areas = fix_metre_areas(area_xy)
    traffic_fixed, bad_hours = fix_missing_traffic(traffic)
    return {
        "grid": grid_fixed,
        "area_xy": xy_fixed,
        "traffic": traffic_fixed,
        "duplicate_days": bad_days,
        "metre_areas": bad_areas,
        "missing_hours": bad_hours,
    }