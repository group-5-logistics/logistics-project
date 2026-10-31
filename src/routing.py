# Analysis: Routing - Segun
# Distances, nearest hub and travel times.

import numpy as np

# to load the file 
path = sys.argv[1] if len(sys.argv) > 1 else "group05_logistics.npz"
d = np.load(path, allow_pickle=True)

# to extract the data
hubs, hub_xy = d["hubs"], d["hub_xy_km"]
areas, area_xy = d["areas"], d["area_xy_km"].copy()
traffic = d["traffic_factor_by_hour"].copy()
first_hour = int(d["first_hour"])

# Distance from every area to every hub in one broadcasting step: (12, 3)
dist = np.linalg.norm(area_xy[:, None, :] - hub_xy[None, :, :], axis=2)
# serving hub index per area
nearest = dist.argmin(axis=1) 
 # km to serving hub
d_min = dist.min(axis=1)              