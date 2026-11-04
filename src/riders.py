"""Calculate rider busy time and rider demand for each hub."""

import numpy as np

MINUTES_PER_HOUR = 60


def busy_minutes(one_way_minutes, handover_minutes):
    """Return how many minutes a rider spends on one delivery."""
    return 2 * np.asarray(one_way_minutes) + handover_minutes


def hub_membership(nearest_hub_index, hub_count=3):
    """Make a table showing which hub serves each area."""
    nearest_hub_index = np.asarray(nearest_hub_index, dtype=int)
    return np.eye(hub_count, dtype=int)[nearest_hub_index]


def riders_needed(orders_grid, busy_time, membership):
    """Return the rider-minutes and riders needed for each day, hour, and hub.

    in:  orders_grid (days, 14, 12), busy_time (14, 12), membership (12, 3)
    out: minutes_per_hub (days, 14, 3), riders (days, 14, 3), nan on copied days
    """
    orders_grid = np.asarray(orders_grid)
    busy_time = np.asarray(busy_time)
    membership = np.asarray(membership)

    # Each order uses a rider for its round trip and handover time.
    rider_minutes = orders_grid * busy_time[None, :, :]    # (days, 14, 12)

    # Add the rider-minutes for areas served by each hub.
    minutes_per_hub = rider_minutes @ membership           # (days, 14, 3)

    # A rider can work 60 minutes in an hour, so round up to whole riders.
    riders = np.ceil(minutes_per_hub / MINUTES_PER_HOUR)
    return minutes_per_hub, riders
