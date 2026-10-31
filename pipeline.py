# Integration - Chudi

'''
SwiftDrop Lagos delivery dispatch pipeline.
Usage: python3 pipeline.py data/group05_logistics.npz
'''

import sys

import numpy as np

from src.structure import load_data, to_grid
from src.problems import find_and_fix, fix_duplicate_days, fix_metre_areas, fix_missing_traffic
from src.routing import distance_matrix, nearest_hub, one_way_minutes
from src.riders import busy_minutes, hub_membership, riders_needed
from src.answers import (riders_per_hub, hub_workload, busiest_hour, longest_trips, make_charts)

def analyse():
    pass


def effect_of_fixes():
    pass


def run_pipeline():
    pass


def print_summary():
    pass


if __name__ == "__main__":
    run_pipeline()