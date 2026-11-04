# Analysis: Routing - Segun
# Distances, nearest hub and travel times.

import numpy as np


def distance_matrix(area_xy, hub_xy):
    """Straight-line distance from every area to every hub.

    in:  area_xy (12, 2) km, hub_xy (3, 2) km
    out: dist (12, 3) km
    """
    # Distance from every area to every hub in one broadcasting step: (12, 3)
    dist = np.linalg.norm(area_xy[:, None, :] - hub_xy[None, :, :], axis=2)
    return dist


def nearest_hub(dist):
    """Each area is served by its nearest hub.

    in:  dist (12, 3)
    out: nearest (12,) hub index per area, d_min (12,) km to that hub
    """
    # serving hub index per area
    nearest = dist.argmin(axis=1)
    # km to serving hub
    d_min = dist.min(axis=1)
    return nearest, d_min


def one_way_minutes(d_min, traffic, params):
    """Minutes to ride from the hub to each area at each hour.

    in:  d_min (12,) km, traffic (14,), params dict
    out: one_way (14, 12) minutes
    """
    free_flow = d_min * params["road_factor"] / params["free_flow_speed_kmh"] * 60   # (12,)
    one_way = traffic[:, None] * free_flow[None, :]                                    # (14, 12)
    return one_way
