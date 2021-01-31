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
    Read in location data from csv and return a pandas dataframe 

        PARAMS:
            filename (str) - imported files
            dtype (list) - column headers
            keep (list) - keep these columns and discard remaining columns
    '''

    df = pd.read_csv(filename, header=0)
    df['Latitude'] = df['Latitude'].replace("–", "", regex=True).astype(float)
    df['Longitude'] = df['Longitude'].replace(
        "–", "", regex=True).astype(float)

    if keep: df = df[keep]
    return df


def load_data(distribution_center_file, population_file):
    '''
    Returns data from files as a pandas dataframe

        PARAMS:
            distribution_center_file (str) - maps latitude and longitude to distribution sites
            population_file (str) - maps latitude and longitude to county
    '''

    county_data = load_csv(
        population_file,
        ['County', 'Latitude', 'Longitude']
    )
    distribution_data = load_csv(
        distribution_center_file,
        ['Name', 'Latitude', 'Longitude']
    )

    return distribution_data, county_data


def build_bqm(potential_new_dist_nodes, num_poi, pois, num_dist, dist_sites, num_new_dist):
    """ 
    Build bqm that models our problem scenario for the hybrid sampler. 

        PARAMS:
            potential_new_dist_nodes (list) - coordinates of potential distribution sites to choose from
            num_poi (int) - number of counties
            pois (list) - coordinates of counties
            num_dist (int) - number of current distribution sites
            dist_sites (list) - coordinates of current distribution sites
            num_new_dist (int) - desired number of distribution sites to add

        RETURNS:
            bqm with optimization parameters set
    """

    # Tunable linear coefficients for optimization problem
    gamma1 = len(potential_new_dist_nodes) * 4
    gamma2 = len(potential_new_dist_nodes) / 3
    gamma3 = len(potential_new_dist_nodes) * 1.7
    gamma4 = len(potential_new_dist_nodes) ** 3

    # Build BQM using adjVectors to find best new distribution sites s.t. min
    # distance to POIs and max distance to existing distribution sites
    bqm = dimod.AdjVectorBQM(len(potential_new_dist_nodes), 'BINARY')

    # Constraint 1: Min average distance to POIs
    if num_poi > 0:
        for i in range(len(potential_new_dist_nodes)):
            # Compute average distance to POIs from this node
            avg_dist = 0
            cand_loc = potential_new_dist_nodes[i]
            for loc in pois:
                dist = (cand_loc[0]**2 - 2*cand_loc[0]*loc[0] + loc[0]**2
                        + cand_loc[1]**2 - 2*cand_loc[1]*loc[1] + loc[1]**2)
                avg_dist += dist / num_poi
            bqm.linear[i] += avg_dist * gamma1

    # Constraint 2: Max distance to other new charging locations
    if num_new_dist > 1:
        for i in range(len(potential_new_dist_nodes)):
            for j in range(i+1, len(potential_new_dist_nodes)):
                ai = potential_new_dist_nodes[i]
                aj = potential_new_dist_nodes[j]
                dist = (-1*ai[0]**2 + 2*ai[0]*aj[0] - aj[0]**2 - ai[1]**2
                        + 2*ai[1]*aj[1] - aj[1]**2)
                bqm.add_interaction(i, j, dist * gamma3)

    # Constraint 3: Choose exactly num_new_dist new charging locations
    bqm.update(dimod.generators.combinations(
        bqm.variables, num_new_dist, strength=gamma4))

    return bqm


def build_graph(population_nodes, distribution_nodes):
    """ Creates the graph of Texas using counties and distribution centers
    as nodes.

        PARAMS:
            population_nodes (list) - coordinates of the counties stored as tuples
            distribution_nodes (list) - coordinates of the distribution centers stored as tuples

        RETURNS:
            Graph G containing all counties and dist centers as nodes with distance
            as edge weights, list of coord of dist sites, and list of cooord of counties
    """
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


def printout_solution_to_cmdline(pois, num_poi, distribution_sites, num_dist, new_dist_nodes, num_new_dist):
    """ 
    Print solution statistics to command line. 

        PARAMS:
            pois (list) - coordinates of counties
            num_poi (int) - number of counties
            dist_sites (list) - coordinates of current distribution sites
            num_dist (int) - number of current distribution sites
            new_dist_nodes (list) - coordinates of potential distribution sites to choose from
            num_new_dist (int) - desired number of distribution sites to add    
    """

    print("\nSolution returned: \n------------------")

    print("\nNew distribution locations:\t\t\t\t", new_dist_nodes)

    if num_poi > 0:
        poi_avg_dist = [0] * len(new_dist_nodes)
        for loc in pois:
            for i, new in enumerate(new_dist_nodes):
                poi_avg_dist[i] += sum(abs(a - b)
                                       for a, b in zip(new, loc)) / num_poi
        print("Average distance to counties:\t\t\t", poi_avg_dist)

    if num_dist > 0:
        old_cs_avg_dist = [sum(abs(a - b) for a, b in zip(new, loc)
                               for loc in distribution_sites) / num_dist for new in new_dist_nodes]
        print("Average distance to old distribution centers:\t", old_cs_avg_dist)

    if num_new_dist > 1:
        new_cs_dist = 0
        for i in range(num_new_dist):
            for j in range(i+1, num_new_dist):
                new_cs_dist += abs(new_dist_nodes[i][0]-new_dist_nodes[j][0])+abs(
                    new_dist_nodes[i][1]-new_dist_nodes[j][1])
        print("Distance between new distribution centers:\t\t\t", new_cs_dist)


def run_bqm_and_collect_solutions(bqm, sampler, potential_new_dist_nodes, **kwargs):
    """ Solve the bqm with the provided sampler to find new charger locations. """

    sampleset = sampler.sample(bqm, **kwargs)

    ss = sampleset.first.sample
    new_dist_nodes = [potential_new_dist_nodes[k]
                          for k, v in ss.items() if v == 1]

    return new_dist_nodes


def iterate_bqm_and_graph(iters, run_num):
    step = 77 // iters
    d, c = load_data('data/Distribution_center_locations_TX.csv', 'data/TX_Counties.csv')
    
    all_nodes = []
    all_pois = []
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

            gmap.apikey = ""

            # scatter method of map object
            # scatter points on the google map
            gmap.scatter(latitude_list, longitude_list, 'blue',
                         size=8000, marker=False)

            gmap.draw("results/run_{}/map_{}.html".format(run_num, i))
        except:
            print("STEP 0")

        all_nodes.append(new_charging_nodes)
        all_pois.append(poi_avg_dist)

    np.savetxt("results/run_{}/sites_{}.txt".format(run_num, i), all_nodes)
    np.savetxt("results/run_{}/distances_{}.txt".format(run_num, i), all_pois)
    
    print("ALL NODES {}".format(all_nodes))
    print("ALL POIS {}".format(all_pois))
        
        

if __name__ == '__main__':
    # Load population 65+, county, and distribution center data from csv
    d, c = load_data(
        'data/Distribution_center_locations_TX.csv',
        'data/TX_Counties.csv'
    )

    # Set the desired number of new distribution centers to add to the map
    num_new_dist = 7

    # Set the desired current distribution centers on the map
    current_dist = []

    # The desired number of current distribution centers on the map
    num_current_dist = len(current_dist)

    # Build large grid graph for city
    G, sites, counties = build_graph(c, d)

    # Build BQM - SAMPLE NEW DIST FROM LIST OF DIST ONLY
    bqm = build_bqm(sites, len(counties), counties, num_current_dist, current_dist, num_new_dist)
    
    # Build BQM - SAMPLE NEW DIST FROM LIST OF COUNTIES ONLY
    # bqm = build_bqm(counties, len(counties), counties, num_current_dist, current_dist, num_new_dist)

    # Run BQM on HSS
    sampler = LeapHybridSampler()
    print("\nRunning scenario on", sampler.solver.id, "solver...")

    new_dist_nodes = run_bqm_and_collect_solutions(bqm, sampler, sites)

    # Print results to commnand-line for user
    printout_solution_to_cmdline(counties, len(
        counties), current_dist, num_current_dist, new_dist_nodes, len(new_dist_nodes))
    
    latitude_list = [x[0] for x in new_dist_nodes]
    longitude_list = [x[1] for x in new_dist_nodes]

    gmap = gmplot.GoogleMapPlotter(latitude_list[0],
                                   longitude_list[0], 1000)

    gmap.apikey = ""

    # scatter method of map object
    # scatter points on the google map
    gmap.scatter(latitude_list, longitude_list, 'blue',
                 size=8000, marker=False)

    gmap.draw("mapNorm_{}.html".format(num_new_dist))