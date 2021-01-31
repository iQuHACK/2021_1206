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
import os
import gmplot

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


def build_bqm(potential_new_cs_nodes, num_poi, pois, num_cs, charging_stations, num_new_cs):
    """ Build bqm that models our problem scenario for the hybrid sampler. """

    # Tunable parameters
    gamma1 = len(potential_new_cs_nodes) * 4
    gamma2 = len(potential_new_cs_nodes) / 3
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

    for i in range(len(population_nodes)):
        # potential distribution nodes
        name1 = population_nodes["County"][i]
        lat1 = float(population_nodes["Latitude"][i])
        lon1 = float(population_nodes["Longitude"][i])
        loc1 = (lat1, lon1)
        sites.append(loc1)
        sites.append(loc1)
        
        G.add_node(name1)
        
        for j in range(i, len(population_nodes)):
            # POIS
            name2 = population_nodes["County"][j]
            lat2 = float(population_nodes["Latitude"][j])
            lon2 = float(population_nodes["Longitude"][j])
            population = float(population_nodes['Number 65+ Population'][j])
            G.add_node(name2)
            
            loc2 = (lat2, lon2)
            weight = great_circle_distance(loc1, loc2) * population
            counties.append(loc2)
            
            
            G.add_edge(name1, name2, weight=weight)
            
    
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
    
    
def iterate_bqm_and_graph(iters, run_num):
    step = 77 // iters
    d, c = load_data('data/Distribution_center_locations_TX.csv', 'data/TX_Counties.csv')
    
    os.mkdir("results/run_{}".format(run_num))

    for i in range(0, 77, step):
        # Build large grid graph for city
        G, sites, counties = build_graph(c, d)

        # Build BQM
        bqm = build_bqm(sites, len(counties), counties, 0, [], i)

        # Run BQM on HSS
        sampler = LeapHybridSampler()
        print("\nRunning scenario on", sampler.solver.id, "solver...")

        new_charging_nodes = run_bqm_and_collect_solutions(bqm, sampler, sites)

        # Print results to commnand-line for user
        poi_avg_dist = printout_solution_to_cmdline(counties, len(counties), [], 0, new_charging_nodes,
                                                    len(new_charging_nodes))

        try:
            latitude_list = [x[0] for x in new_charging_nodes]
            longitude_list = [x[1] for x in new_charging_nodes]

            gmap = gmplot.GoogleMapPlotter(latitude_list[0],
                                           longitude_list[0], 1000)

            gmap.apikey = "AIzaSyB9N6G3mW559tfaPnaI_QVJo5MaTiwtOkE"

            # scatter method of map object
            # scatter points on the google map
            gmap.scatter(latitude_list, longitude_list, 'blue',
                         size=8000, marker=False)

#             gmap.draw("map_{}.html".format(run_num, i))
            
        except:
            print("STEP 0")

        print("charging_nodes: {}".format(new_charging_nodes))
        print("poi_distance: {}".format(poi_avg_dist))
        
        

    
if __name__ == '__main__':
    d, c = load_data('data/Distribution_center_locations_TX.csv', 'data/TX_Counties.csv')
#     d = add_coords(d)
#     c = add_coords(c)
    
    # Build large grid graph for city
    G, sites, counties = build_graph(c, d)
    
    # Build BQM
    n = 70
    bqm = build_bqm(sites, len(sites), counties, 0, [], n)

    # Run BQM on HSS
    sampler = LeapHybridSampler()
    print("\nRunning scenario on", sampler.solver.id, "solver...")
    
    new_charging_nodes = run_bqm_and_collect_solutions(bqm, sampler, sites)

    # Print results to commnand-line for user
    printout_solution_to_cmdline(sites, len(sites), [], 0, new_charging_nodes, len(new_charging_nodes))
    
    latitude_list = [x[0] for x in new_charging_nodes]
    longitude_list = [x[1] for x in new_charging_nodes]

    gmap = gmplot.GoogleMapPlotter(latitude_list[0],
                                   longitude_list[0], 10000)

    gmap.apikey = "AIzaSyB9N6G3mW559tfaPnaI_QVJo5MaTiwtOkE"

    # scatter method of map object
    # scatter points on the google map
    gmap.scatter(latitude_list, longitude_list, 'blue',
                 size=8000, marker=False)
    
    gmap.draw("map_{}.html".format(n))
    
    

    # Create scenario output image
#     save_output_image(G, pois, charging_stations, new_charging_nodes)

    