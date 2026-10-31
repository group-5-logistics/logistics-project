"""Calculate rider busy time and rider demand for each hub."""

import numpy as np


def busy_minutes(one_way_minutes, handover_minutes):
    """Return how many minutes a rider spends on one delivery."""
    return 2 * np.asarray(one_way_minutes) + handover_minutes


def hub_membership(nearest_hub_index, hub_count=3):
    """Make a table showing which hub serves each area."""
    nearest_hub_index = np.asarray(nearest_hub_index, dtype=int)
    return np.eye(hub_count, dtype=int)[nearest_hub_index]


def riders_needed(orders_grid, busy_time, membership):
    """Return the riders needed for each day, hour, and hub."""
    orders_grid = np.asarray(orders_grid)
    busy_time = np.asarray(busy_time)
    membership = np.asarray(membership)

    # Each order uses a rider for its round trip and handover time.
    rider_minutes = orders_grid * busy_time[None, :, :]

    # Add the rider-minutes for areas served by each hub.
    hub_count = membership.shape[1]
    minutes_per_hub = np.zeros(
        (orders_grid.shape[0], orders_grid.shape[1], hub_count)
    )
    for hub in range(hub_count):
        minutes_per_hub[:, :, hub] = np.sum(
            rider_minutes * membership[None, None, :, hub], axis=2
        )

    # A rider can work 60 minutes in an hour, so round up to whole riders.
    return np.ceil(minutes_per_hub / 60).astype(int)
