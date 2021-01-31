import random
import argparse
import dimod
import sys
import networkx as nx
import numpy as np
import pandas as pd
from dwave.system import LeapHybridSampler
from utils import *

import matplotlib
try:
    import matplotlib.pyplot as plt
except ImportError:
    matplotlib.use("agg")
    import matplotlib.pyplot as plt


def load_csv(filename, dtype=None, keep=None):
    '''
    Returns a pandas df from a csv file. keep (list) contains columns to keep.
    '''
    df = pd.read_csv(filename, header=0)
    df['Latitude'] = df['Latitude'].replace("–","", regex=True).astype(float)
    df['Longitude'] = df['Longitude'].replace("–","", regex=True).astype(float)
    if keep:
        df = df[keep]
    return df


def load_data(distribution_center_file, county_file):
    '''
    Returns data from files as pandas df
    '''
    
    distribution_data = load_csv(distribution_center_file, ['Latitude', 'Longitude'])
    county_data = load_csv(county_file, ['County [2]', 'Latitude', 'Longitude', 'Number 65+ Population'])
    
    return distribution_data, county_data


def add_coords(df):
    '''
    Adds new column, result of merging Latitude and Longitude columns into one column called coordinates
    '''
    coords_col = []
    for _, row in df.iterrows():
        coords = (row['Latitude'], row['Longitude'])
        coords_col.append(coords)
    df['coordinates'] = coords_col
    
    return df


def find_closest_center(distribution_data, county_data):
    '''
    Ranks the closest vaccine distribution center to a zip code by distance
    
    Returns a np array with index i as zipcode i. array[i] is an array of tuples (center j, dist to center j from zipcode i)
    '''
    county_to_closest_center = np.empty(len(county_data))
    
    for county_i, county_row in county_data.iterrows():
        # get county data
        county, num, lat, lon = county_row['County [2]'], county_row['Number 65+ Population'], county_row['Latitude'], county_row['Longitude']
        ranked = []
        for _, d_row in distribution_data.iterrows():
            d_lat, d_lon = d_row['Latitude'], d_row['Longitude']
            dist = great_circle_distance((lat, lon), (d_lat, d_lon))
            ranked.append((d_i, dist))
        ranked.sort(key=lambda x: x[1]) # sort by centers by distance
        ranked = np.array(ranked)
        county_to_closest_center[county_i] = ranked # save this zipcode's closest centers
    
    return county_to_closest_center


def build_bqm(num_poi, pois, num_of_dcenters, distribution_centers, vaccines):
    """ Build bqm that models our problem scenario for the hybrid sampler. """

    # Tunable parameters
    gamma1 = num_of_dcenters * 4
#     gamma2 = len(potential_new_cs_nodes) / 3
    gamma4 = num_of_dcenters ** 3

    # Build BQM using adjVectors to find best new charging location s.t. min distance to POIs and max distance to existing charging locations
    bqm = dimod.AdjVectorBQM(num_of_dcenters, 'BINARY')

    # Constraint 1: Min average distance to POIs
    for i in range(len(distribution_centers)):
        # Compute average distance to POIs from this node
        dc = distribution_centers[i]
        avg_dist = 0
        for loc in pois:
            dist = (dc[0]**2 - 2*dc[0]*loc[0] + loc[0]**2 + dc[1]**2 - 2*dc[1]*loc[1] + loc[1]**2)
            avg_dist += dist / num_poi 
        bqm.linear[i] += avg_dist * gamma1

    # Constraint 2: Max distance to existing chargers 

    # Constraint 4: Allocate exactly num of total vaccines
    bqm.update(dimod.generators.combinations(bqm.variables, num_of_dcenters, strength=gamma4))

    return bqm


def run_bqm(bqm, sampler, **kwargs):
    """ 
    Solve the bqm with the provided sampler
    """

    sampleset = sampler.sample(bqm, **kwargs)

    ss = sampleset.first.sample
    new_charging_nodes = [potential_new_cs_nodes[k] for k, v in ss.items() if v == 1]

    return new_charging_nodes


if __name__ == '__main__':
    
    # From the data, there have been 2 * (1,238,250 + 1,271,400) or 5,019,300 vaccines allocated to TX since 12/14/2020
    
    total_vaccines = 4042762 # actually total # of ppl 65+
    
    # load data
    d, c = load_data('data/Distribution_center_locations_TX.csv', 'data/TX_Counties.csv')
    d = add_coords(d)
    c = add_coords(c)
    
    num_of_pois = len(c)
    num_of_dcenters = len(d)
    
    pois = list(c['coordinates'])
    distribution_centers = list(d['coordinates'])
    
    # Build BQM
    bqm = build_bqm(num_of_pois, pois, num_of_dcenters, distribution_centers, total_vaccines)
    
    
    
    sampler = LeapHybridSampler()
#     results = run_bqm_and_collect_solutions(bqm, sampler)
    
    
    