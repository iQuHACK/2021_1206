import random
import argparse
import dimod
import sys
import networkx as nx
import numpy as np
import pandas as pd
from dwave.system import LeapHybridSampler
from util import great_circle_distance
import pylab

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

def load_data(distribution_center_file, population_file):

    '''
    Returns data from files as pandas df
    '''
    county_data = load_csv(population_file, ['County','Latitude', 'Longitude'])
    distribution_data = load_csv(distribution_center_file, ['Name','Latitude', 'Longitude'])
    
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

def build_bqm(potential_new_cs_nodes, num_poi, pois, num_cs, charging_stations, num_new_cs):
    """ Build bqm that models our problem scenario for the hybrid sampler. """

    # Tunable parameters
    gamma1 = len(potential_new_cs_nodes) * 4
    gamma2 = len(potential_new_cs_nodes) / 3
    gamma3 = len(potential_new_cs_nodes) * 1.7
    gamma4 = len(potential_new_cs_nodes) ** 3

    # Build BQM using adjVectors to find best new charging location s.t. min 
    # distance to POIs and max distance to existing charging locations
    bqm = dimod.AdjVectorBQM(len(potential_new_cs_nodes), 'BINARY')

    # Constraint 1: Min average distance to POIs
    if num_poi > 0:
        for i in range(len(potential_new_cs_nodes)):
            # Compute average distance to POIs from this node
            avg_dist = 0
            cand_loc = potential_new_cs_nodes[i]
            for loc in pois:
                dist = (cand_loc[0]**2 - 2*cand_loc[0]*loc[0] + loc[0]**2 
                                    + cand_loc[1]**2 - 2*cand_loc[1]*loc[1] + loc[1]**2)
                avg_dist += dist / num_poi 
            bqm.linear[i] += avg_dist * gamma1

    # Constraint 2: Max distance to existing chargers
    if num_cs > 0:
        for i in range(len(potential_new_cs_nodes)):
            # Compute average distance to POIs from this node
            avg_dist = 0
            cand_loc = potential_new_cs_nodes[i]
            for loc in charging_stations:
                dist = (-1*cand_loc[0]**2 + 2*cand_loc[0]*loc[0] - loc[0]**2
                                    - cand_loc[1]**2 + 2*cand_loc[1]*loc[1] - loc[1]**2)
                avg_dist += dist / num_cs
            bqm.linear[i] += avg_dist * gamma2

    # Constraint 3: Max distance to other new charging locations
    if num_new_cs > 1:
        for i in range(len(potential_new_cs_nodes)):
            for j in range(i+1, len(potential_new_cs_nodes)):
                ai = potential_new_cs_nodes[i]
                aj = potential_new_cs_nodes[j]
                dist = (-1*ai[0]**2 + 2*ai[0]*aj[0] - aj[0]**2 - ai[1]**2 
                        + 2*ai[1]*aj[1] - aj[1]**2)
                bqm.add_interaction(i, j, dist * gamma3)

    # Constraint 4: Choose exactly num_new_cs new charging locations
    bqm.update(dimod.generators.combinations(bqm.variables, num_new_cs, strength=gamma4))

    return bqm

def run_bqm(bqm, sampler, **kwargs):
    """ 
    Solve the bqm with the provided sampler
    """

    sampleset = sampler.sample(bqm, **kwargs)

    ss = sampleset.first.sample
    new_charging_nodes = [potential_new_cs_nodes[k] for k, v in ss.items() if v == 1]

    return new_charging_nodes

def build_graph(population_nodes, distribution_nodes):
    G = nx.Graph()
    sites = []
    counties = []

    for i in range(len(distribution_nodes)):
        name1 = distribution_nodes["Name"][i]
        lat1 = float(distribution_nodes["Latitude"][i])
        lon1 = float(distribution_nodes["Longitude"][i])
        loc1 = (lat1, lon1)
        sites.append(loc1)
        
        G.add_node(name1)
        
        
        
        for j in range(len(population_nodes)):
            name2 = population_nodes["County"][j]
            lat2 = float(population_nodes["Latitude"][j])
            lon2 = float(population_nodes["Longitude"][j])
            G.add_node(name2)
            
            loc2 = (lat2, lon2)
            counties.append(loc2)
            
            G.add_edge(name1, name2, weight=great_circle_distance(loc1, loc2))
            
    
    return G, sites, counties

def printout_solution_to_cmdline(pois, num_poi, charging_stations, num_cs, new_charging_nodes, num_new_cs):
    """ Print solution statistics to command line. """

    print("\nSolution returned: \n------------------")
    
    print("\nNew charging locations:\t\t\t\t", new_charging_nodes)

    if num_poi > 0:
        poi_avg_dist = [0] * len(new_charging_nodes)
        for loc in pois:
            for i, new in enumerate(new_charging_nodes):
                poi_avg_dist[i] += sum(abs(a - b) for a, b in zip(new, loc)) / num_poi
        print("Average distance to POIs:\t\t\t", poi_avg_dist)

    if num_cs > 0:
        old_cs_avg_dist = [sum(abs(a - b) for a, b in zip(new, loc) for loc in charging_stations) / num_cs for new in new_charging_nodes]
        print("Average distance to old charging stations:\t", old_cs_avg_dist)

    if num_new_cs > 1:
        new_cs_dist = 0
        for i in range(num_new_cs):
            for j in range(i+1, num_new_cs):
                new_cs_dist += abs(new_charging_nodes[i][0]-new_charging_nodes[j][0])+abs(new_charging_nodes[i][1]-new_charging_nodes[j][1])
        print("Distance between new chargers:\t\t\t", new_cs_dist)
        
def run_bqm_and_collect_solutions(bqm, sampler, potential_new_cs_nodes, **kwargs):
    """ Solve the bqm with the provided sampler to find new charger locations. """

    sampleset = sampler.sample(bqm, **kwargs)

    ss = sampleset.first.sample
    new_charging_nodes = [potential_new_cs_nodes[k] for k, v in ss.items() if v == 1]

    return new_charging_nodes
    

    
if __name__ == '__main__':
    d, c = load_data('data/Distribution_center_locations_TX.csv', 'data/TX_Counties.csv')
#     d = add_coords(d)
#     c = add_coords(c)
    
    # Build large grid graph for city
    G, sites, counties = build_graph(c, d)
    
    # Build BQM
    bqm = build_bqm(sites, len(sites), counties, 0, [], 10)

    # Run BQM on HSS
    sampler = LeapHybridSampler()
    print("\nRunning scenario on", sampler.solver.id, "solver...")
    
    new_charging_nodes = run_bqm_and_collect_solutions(bqm, sampler, sites)

    # Print results to commnand-line for user
    printout_solution_to_cmdline(sites, len(sites), [], 0, new_charging_nodes, len(new_charging_nodes))

    # Create scenario output image
#     save_output_image(G, pois, charging_stations, new_charging_nodes)