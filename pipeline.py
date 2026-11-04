# Integration - Chudi

'''
SwiftDrop Lagos delivery dispatch pipeline.
Usage: python3 pipeline.py data/group05_logistics.npz
'''

import sys

import numpy as np

from src.structure import load_data, to_grid
from src.problems import find_and_fix
from src.routing import distance_matrix, nearest_hub, one_way_minutes
from src.riders import busy_minutes, hub_membership, riders_needed
from src.answers import (riders_per_hub, hub_workload, busiest_hour, longest_trips, make_charts)


def analyse(grid, area_xy, hub_xy, traffic, params):
    """Distances, trip times and riders for one version of the data.

    in:  grid (days, 14, 12), area_xy (12, 2), hub_xy (3, 2), traffic (14,), params dict
    out: dict of the arrays below
    """
    dist = distance_matrix(area_xy, hub_xy)                          # (12, 3)
    nearest, d_min = nearest_hub(dist)                               # (12,), (12,)
    one_way = one_way_minutes(d_min, traffic, params)                # (14, 12)
    busy = busy_minutes(one_way, params["handover_minutes"])         # (14, 12)
    membership = hub_membership(nearest, hub_count=len(hub_xy))      # (12, 3)
    rider_minutes, riders = riders_needed(grid, busy, membership)    # (days, 14, 3) each
    return {
        "dist": dist, "nearest": nearest, "one_way": one_way, "busy": busy,
        "membership": membership, "rider_minutes": rider_minutes, "riders": riders,
        "p95_riders": riders_per_hub(riders, params),                # (3,)
    }


def fix_effect(result, missing_hours):
    """Numbers to compare one run of analyse with another.

    in:  result dict from analyse, missing_hours (k,)
    out: p95 riders (3,), mean riders per hour (3,), mean riders in the filled hours (3,)
    """
    riders = result["riders"]
    mean_riders = np.nanmean(riders, axis=(0, 1))
    if missing_hours.size == 0:
        return result["p95_riders"], mean_riders, np.full(riders.shape[2], np.nan)
    at_missing = np.nanmean(riders[:, missing_hours, :], axis=(0, 1))
    return result["p95_riders"], mean_riders, at_missing


def effect_of_fixes(grid, raw, clean, fixed_result):
    """Run the analysis again with each fix switched off, one at a time.

    in:  grid (days, 14, 12) before fixing, raw dict from load_data,
         clean dict from find_and_fix, fixed_result from analyse on the clean data
    out: dict of scenario -> fix_effect numbers
    """
    params, hub_xy, missing = raw["params"], raw["hub_xy_km"], clean["missing_hours"]

    # A careless fix: treat the missing hour as if there were no traffic
    traffic_as_one = raw["traffic_factor_by_hour"].copy()
    traffic_as_one[missing] = 1.0

    day_kept = analyse(grid, clean["area_xy"], hub_xy, clean["traffic"], params)
    metres_kept = analyse(clean["grid"], raw["area_xy_km"], hub_xy, clean["traffic"], params)
    traffic_one = analyse(clean["grid"], clean["area_xy"], hub_xy, traffic_as_one, params)
    return {
        "all fixed": fix_effect(fixed_result, missing),
        "copied day kept": fix_effect(day_kept, missing),
        "area left in metres": fix_effect(metres_kept, missing),
        "missing traffic set to 1.0": fix_effect(traffic_one, missing),
    }


def run_pipeline(path, out_dir="figures", verbose=True):
    """Run every stage on one data file and save the charts.

    in:  path to a group05 style .npz file, folder for the charts
    out: dict with every result
    """
    raw = load_data(path)
    params, first_hour = raw["params"], raw["first_hour"]
    hubs, areas = raw["hubs"], raw["areas"]
    traffic = raw["traffic_factor_by_hour"]

    grid = to_grid(raw["orders"], areas.size, traffic.size)          # (56, 14, 12)
    clean = find_and_fix(grid, raw["area_xy_km"], traffic)
    res = analyse(clean["grid"], clean["area_xy"], raw["hub_xy_km"], clean["traffic"], params)

    orders_per_day, avg_busy = hub_workload(clean["grid"], res["busy"], res["membership"])
    orders_per_hour, busiest_orders, busiest_workload = busiest_hour(
        clean["grid"], res["rider_minutes"], first_hour)
    max_per_area, longest_hour_idx, longest_area_idx, misses = longest_trips(res["one_way"], params)

    make_charts(out_dir, hubs, areas, first_hour, res["p95_riders"], orders_per_hour,
                res["riders"], max_per_area, params)

    # everything from analyse, plus the answers
    results = {
        **res,
        "raw": raw, "grid": grid, "clean": clean,
        "effects": effect_of_fixes(grid, raw, clean, res),
        "orders_per_day": orders_per_day, "avg_busy": avg_busy,
        "busiest_orders": busiest_orders, "busiest_workload": busiest_workload,
        "max_per_area": max_per_area, "longest_hour_idx": longest_hour_idx,
        "longest_area_idx": longest_area_idx, "misses": misses,
    }
    if verbose:
        print_summary(results)
    return results


def print_summary(results):
    """Print the evidence for each data problem, then the answers."""
    raw, clean = results["raw"], results["clean"]
    hubs, areas, params = raw["hubs"], raw["areas"], raw["params"]
    first_hour = raw["first_hour"]

    print("SHAPES")
    print(f"  orders     {raw['orders'].shape}")
    print(f"  grid       {results['grid'].shape}")
    print(f"  distances  {results['dist'].shape}")
    print(f"  one_way    {results['one_way'].shape}")
    print(f"  riders     {results['riders'].shape}")
    print(f"  per hub    {results['p95_riders'].shape}")

    print("\nDATA PROBLEMS")
    copied_days = clean["duplicate_days"]
    if copied_days.size == 0:
        print("  1. No copied days found")
    for d in copied_days:
        print(f"  1. Day {d} is a copy of day {d - 1} ({results['grid'][d].sum()} orders each) -> set to nan")

    metre_areas = clean["metre_areas"]
    if metre_areas.size == 0:
        print("  2. No areas in metres found")
    for a in metre_areas:
        x_before, y_before = raw["area_xy_km"][a]
        x_after, y_after = clean["area_xy"][a]
        print(f"  2. {areas[a]} in metres: ({x_before:.1f}, {y_before:.1f}) -> ({x_after:.3f}, {y_after:.3f}) km")

    missing_hours = clean["missing_hours"]
    if missing_hours.size == 0:
        print("  3. No missing traffic factors found")
    for h in missing_hours:
        print(f"  3. Traffic factor missing at {first_hour + h:02d}:00 -> filled with {clean['traffic'][h]:.3f}")

    print(f"\nQ1 RIDERS PER HUB ({100 * params['target_share']:.0f}th percentile of all hours)")
    for h, name in enumerate(hubs):
        served = ", ".join(areas[results["nearest"] == h])
        print(f"  {name:<6} {results['p95_riders'][h]:>3} riders | {results['orders_per_day'][h]:.0f} orders/day | "
              f"{results['avg_busy'][h]:.1f} busy min/order | serves: {served}")
    print(f"  Total  {results['p95_riders'].sum():>3} riders")

    print("\nEFFECT OF EACH FIX (per hub: 95th pct riders | mean riders per hour | mean riders in the filled hour)")
    for scenario, (p95, mean_riders, at_missing) in results["effects"].items():
        print(f"  {scenario:<28} {p95.tolist()} | {mean_riders.round(2).tolist()} | {at_missing.round(2).tolist()}")

    print("\nQ2 HARDEST TIMES AND PLACES")
    print(f"  Busiest hour by orders:   {results['busiest_orders']:02d}:00")
    print(f"  Busiest hour by workload: {results['busiest_workload']:02d}:00")
    h, a = results["longest_hour_idx"], results["longest_area_idx"]
    print(f"  Longest one-way trip: {areas[a]} at {first_hour + h:02d}:00, {results['max_per_area'][a]:.1f} min")
    misses = results["misses"]
    if misses.size == 0:
        print(f"  Every area is within {params['target_minutes']} min at every hour.")
    for h, a in misses:
        print(f"  Misses the {params['target_minutes']} min target: {areas[a]} at {first_hour + h:02d}:00")


if __name__ == "__main__":
    run_pipeline(sys.argv[1] if len(sys.argv) > 1 else "data/group05_logistics.npz")
