# Answers - Voke
# Business answers and charts

import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def riders_per_hub(riders, params):
    """Q1: riders each hub needs to cover 95% of all day-hour slots.

    in:  riders (days, 14, 3), params dict
    out: p95_riders (3,) whole riders
    """
    # One row per day-hour slot; nanpercentile skips the copied day
    all_hours = riders.reshape(-1, riders.shape[2])                    # (784, 3)
    p95 = np.nanpercentile(all_hours, 100 * params["target_share"], axis=0)

    # Round up to whole riders
    p95_riders = np.ceil(p95).astype(int)
    return p95_riders


def hub_workload(grid, busy, membership):
    """Why one hub needs more riders: its orders per day and minutes per order.

    in:  grid (days, 14, 12), busy (14, 12), membership (12, 3)
    out: orders_per_day (3,), avg_busy (3,) minutes per order
    """
    orders_per_area = np.nansum(grid, axis=(0, 1))                     # (12,)
    minutes_per_area = np.nansum(grid * busy[None], axis=(0, 1))       # (12,)
    n_days = np.sum(~np.isnan(grid).all(axis=(1, 2)))                  # copied days not counted

    orders_per_day = (orders_per_area @ membership) / n_days           # (3,)
    avg_busy = (minutes_per_area @ membership) / (orders_per_area @ membership)
    return orders_per_day, avg_busy


def busiest_hour(grid, rider_minutes, first_hour):
    """Q2: busiest clock hour by orders and by rider workload.

    in:  grid (days, 14, 12), rider_minutes (days, 14, 3), first hour of the day
    out: orders_per_hour (14,) average orders, busiest hour by orders,
         busiest hour by workload
    """
    # A copied day sums to nan, so nanmean leaves it out
    orders_per_hour = np.nanmean(np.sum(grid, axis=2), axis=0)                # (14,)
    workload_per_hour = np.nanmean(np.sum(rider_minutes, axis=2), axis=0)     # (14,)

    busiest_clock_hour = first_hour + int(np.argmax(orders_per_hour))
    busiest_workload_hour = first_hour + int(np.argmax(workload_per_hour))
    return orders_per_hour, busiest_clock_hour, busiest_workload_hour


def longest_trips(one_way, params):
    """Q2: longest one-way trip, and every trip that misses the target.

    in:  one_way (14, 12), params dict
    out: max_per_area (12,), longest_hour_idx, longest_area_idx,
         misses (k, 2) hour and area index of each trip over target_minutes
    """
    # Find the longest trip in each area
    max_per_area = np.max(one_way, axis=0)                             # (12,)

    # Identify the area and hour of the longest trip
    longest_area_idx = int(np.argmax(max_per_area))
    longest_hour_idx = int(np.argmax(one_way[:, longest_area_idx]))

    # Check which trips miss the target
    misses = np.argwhere(one_way > params["target_minutes"])           # (k, 2)
    return max_per_area, longest_hour_idx, longest_area_idx, misses


def make_charts(out_dir, hubs, areas, first_hour, p95_riders, orders_per_hour,
                riders, max_per_area, params):
    """Save the three report charts as PNG files in out_dir."""
    os.makedirs(out_dir, exist_ok=True)
    hours = np.arange(first_hour, first_hour + orders_per_hour.size)   # (14,)

    # Chart 1: Riders needed per hub
    plt.figure(figsize=(8, 5))
    bars = plt.bar(hubs, p95_riders)
    plt.bar_label(bars)
    plt.title(f"Riders Needed per Hub at {100 * params['target_share']:.0f}th Percentile")
    plt.xlabel("Delivery Hub")
    plt.ylabel("Number of Riders")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "riders_per_hub.png"), dpi=150)
    plt.close()

    # Chart 2: Average riders per hub by hour, orders as faint bars
    plt.figure(figsize=(9, 5))
    plt.bar(hours, orders_per_hour, color="lightgrey")
    plt.xlabel("Hour of Day")
    plt.ylabel("Average Orders (all areas)")
    plt.xticks(hours)
    plt.twinx()
    plt.plot(hours, np.nanmean(riders, axis=0), marker="o")           # one line per hub
    plt.ylim(bottom=0)
    plt.ylabel("Average Riders Needed")
    plt.legend(hubs, loc="upper left")
    plt.title("Orders and Riders Needed by Hour")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "riders_by_hour.png"), dpi=150)
    plt.close()

    # Chart 3: Longest one-way trip per area
    plt.figure(figsize=(11, 5))
    plt.bar(areas, max_per_area)
    plt.title("Longest One-Way Trip per Area")
    plt.xlabel("Delivery Area")
    plt.ylabel("Travel Time (Minutes)")
    plt.xticks(rotation=45)
    plt.axhline(
        y=params["target_minutes"],
        linestyle="--",
        color="red",
        label="Target travel time"
    )
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "longest_trip_per_area.png"), dpi=150)
    plt.close()
