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

def load_data(population_file, distribution_center_file, county_file=''):
    '''
    Returns data from files as pandas csv
    '''
    
    population_data = load_csv(population_file, ['COUNTY','Number 65+ Population'])
    distribution_data = load_csv(distribution_center_file, ['Latitude', 'Longitude'])
#     county_data = load_csv(county_file, ['County', 'Latitude', 'Longitude'])
    
    return population_data, distribution_data

if __name__ == '__main__':
    
    p, d = load_data('Texas_At_Risk_AllData.csv','Distribution_center_locations_TX.csv')
    print(d.head())
    
    
    