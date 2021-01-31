import random
import argparse
import dimod
import sys
import networkx as nx
import numpy as np
import pandas as pd
from dwave.system import LeapHybridSampler
<<<<<<< HEAD
from utils import *
=======
from util import great_circle_distance
import pylab
>>>>>>> 95669390d3cce2ff366b8959b99b9035c6c00d38

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

<<<<<<< HEAD

def load_data(distribution_center_file, county_file):
=======
def load_data(population_file, distribution_center_file):
>>>>>>> 95669390d3cce2ff366b8959b99b9035c6c00d38
    '''
    Returns data from files as pandas df
    '''
    
<<<<<<< HEAD
    distribution_data = load_csv(distribution_center_file, ['Latitude', 'Longitude'])
    county_data = load_csv(county_file, ['County [2]', 'Latitude', 'Longitude', 'Number 65+ Population'])
=======
    population_data = load_csv(population_file, ['County','Latitude', 'Longitude'])
    distribution_data = load_csv(distribution_center_file, ['Name','Latitude', 'Longitude'])
>>>>>>> 95669390d3cce2ff366b8959b99b9035c6c00d38
    
    return distribution_data, county_data

<<<<<<< HEAD

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
=======
<<<<<<< HEAD

def build_graph(population_nodes, distribution_nodes):
    G = nx.Graph()

    for i in range(len(distribution_nodes)):
        name1 = distribution_nodes["Name"][i]
        lat1 = float(distribution_nodes["Latitude"][i])
        lon1 = float(distribution_nodes["Longitude"][i])
        loc1 = (lat1, lon1)
        
        G.add_node(name1)
        
        for j in range(len(population_nodes)):
            name2 = population_nodes["County"][j]
            lat2 = float(population_nodes["Latitude"][j])
            lon2 = float(population_nodes["Longitude"][j])
            G.add_node(name2)
            
            loc2 = (lat2, lon2)
            
            G.add_edge(name1, name2, weight=great_circle_distance(loc1, loc2))
            
    return G
    
    
    

=======
>>>>>>> fb455fd7c13d7300e146f7c703d42144e2bd3b8b
def build_bqm(potential_new_cs_nodes, num_distribution, distribution_data, num_cs, charging_stations, num_new_cs):
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
>>>>>>> 95669390d3cce2ff366b8959b99b9035c6c00d38


if __name__ == '__main__':
    
<<<<<<< HEAD
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
=======
    p, d = load_data('data/TX_Counties.csv','data/Distribution_center_locations_TX.csv')
    G = build_graph(p, d)
    
    pos = nx.spring_layout(G)
    nx.draw(G, pos, node_size=100, node_color='yellow', font_size=8, font_weight='bold')

    plt.tight_layout()
    plt.savefig("Graph.png", format="PNG")
    plt.show()

>>>>>>> 95669390d3cce2ff366b8959b99b9035c6c00d38
    
    
    