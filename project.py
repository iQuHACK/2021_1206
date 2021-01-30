import random
import argparse
import dimod
import sys
import networkx as nx
import numpy as np
import pandas as pd
from dwave.system import LeapHybridSampler

import matplotlib
try:
    import matplotlib.pyplot as plt
except ImportError:
    matplotlib.use("agg")
    import matplotlib.pyplot as plt

def load_csv(filename, keep=None):
    '''
    Returns a pandas df from a csv file. keep (list) contains columns to keep.
    '''
    df = pd.read_csv(filename, header=0)
    if keep:
        df = df[keep]
    return df

def load_data(population_file, distribution_center_file, county_file):
    '''
    Returns data from files as pandas csv
    '''
    
    population_data = load_csv(population_file, ['COUNTY','Number 65+ Population'])
    distribution_data = load_csv(distribution_center_file, ['Latitude', 'Longitude'])
    zipcode_data = load_csv(distribution_center_file, ['Zip', 'Latitude', 'Longitude'])
#     county_data = load_csv(county_file, ['County', 'Latitude', 'Longitude'])
    
    return population_data, distribution_data

# def build_bqm(potential_new_cs_nodes, num_distribution, distribution_data, num_cs, charging_stations, num_new_cs):
#     """ Build bqm that models our problem scenario for the hybrid sampler. """

#     # Tunable parameters
#     gamma1 = len(potential_new_cs_nodes) * 4
#     gamma2 = len(potential_new_cs_nodes) / 3
#     gamma3 = len(potential_new_cs_nodes) * 1.7
#     gamma4 = len(potential_new_cs_nodes) ** 3

#     # Build BQM using adjVectors to find best new charging location s.t. min 
#     # distance to POIs and max distance to existing charging locations
#     bqm = dimod.AdjVectorBQM(len(potential_new_cs_nodes), 'BINARY')

#     # Constraint 1: Min average distance to POIs
#     if num_poi > 0:
#         for i in range(len(potential_new_cs_nodes)):
#             # Compute average distance to POIs from this node
#             avg_dist = 0
#             cand_loc = potential_new_cs_nodes[i]
#             for loc in pois:
#                 dist = (cand_loc[0]**2 - 2*cand_loc[0]*loc[0] + loc[0]**2 
#                                     + cand_loc[1]**2 - 2*cand_loc[1]*loc[1] + loc[1]**2)
#                 avg_dist += dist / num_poi 
#             bqm.linear[i] += avg_dist * gamma1

#     # Constraint 2: Max distance to existing chargers
#     if num_cs > 0:
#         for i in range(len(potential_new_cs_nodes)):
#             # Compute average distance to POIs from this node
#             avg_dist = 0
#             cand_loc = potential_new_cs_nodes[i]
#             for loc in charging_stations:
#                 dist = (-1*cand_loc[0]**2 + 2*cand_loc[0]*loc[0] - loc[0]**2
#                                     - cand_loc[1]**2 + 2*cand_loc[1]*loc[1] - loc[1]**2)
#                 avg_dist += dist / num_cs
#             bqm.linear[i] += avg_dist * gamma2

#     # Constraint 3: Max distance to other new charging locations
#     if num_new_cs > 1:
#         for i in range(len(potential_new_cs_nodes)):
#             for j in range(i+1, len(potential_new_cs_nodes)):
#                 ai = potential_new_cs_nodes[i]
#                 aj = potential_new_cs_nodes[j]
#                 dist = (-1*ai[0]**2 + 2*ai[0]*aj[0] - aj[0]**2 - ai[1]**2 
#                         + 2*ai[1]*aj[1] - aj[1]**2)
#                 bqm.add_interaction(i, j, dist * gamma3)

#     # Constraint 4: Choose exactly num_new_cs new charging locations
#     bqm.update(dimod.generators.combinations(bqm.variables, num_new_cs, strength=gamma4))

#     return bqm


if __name__ == '__main__':
    
    p, d = load_data('data/Texas_At_Risk_AllData.csv','data/Distribution_center_locations_TX.csv')
    print(d.head())
    
    
    