#!/usr/bin/env python
# coding: utf-8

# In[1]:

import scipy.io as sio
import numpy as np
import os
import sys
import argparse
import csv
os.chdir('../../')
sys.path.append('.')
from pyMCDS import pyMCDS
from multiprocessing import Pool
import pickle

parser = argparse.ArgumentParser(description='Process input')
parser.add_argument('--folder', type=str, default="", help='Choose which results to analyse')
parser.add_argument('--replicates', type=int, default=12, help='Inform how many replicated where done')
parser.add_argument('--cores', type=int, default=10, help='Number of cores to use for analysing results')

args = parser.parse_args()

def get_replicate(path, k):
    str_name = 'output{:08d}.xml'.format(k)
        
    mcds = pyMCDS(str_name, path)  
    
    type_by_id = dict(zip(
        mcds.data['discrete_cells']['ID'].astype(int), 
        mcds.data['discrete_cells']['cell_type'].astype(int)
    ))
    types = mcds.data['discrete_cells']['cell_type'].astype(int)

    list_types = list(set(types))
    
    count_dict = {}
    for celltype in list_types:
        if celltype not in count_dict.keys():
            count_dict.update({celltype: {}})
    
    with open(os.path.join(path, 'states_%08u.csv' % k), newline='') as csvfile:
        states_reader = csv.reader(csvfile, delimiter=',')
    
        for row in states_reader:
            if row[0] != 'ID':
                t_id = int(row[0])
                state = row[1]

                if state not in count_dict[type_by_id[t_id]]:
                    count_dict[type_by_id[t_id]][state] = 1
                else:
                    count_dict[type_by_id[t_id]][state] += 1
    
    return count_dict
    
def get_timestep(k, replicates):
    print(".", end = '')
    counts = []
    for j in replicates:
        if len(args.folder) > 0:
            path = os.path.join(args.folder, 'output_R'+str("%02d"%j))
        else:
            path = 'output_R'+str("%02d"%j)
            
        counts.append(get_replicate(path, k))
        
    cell_types = set()
    for count in counts:
        cell_types = cell_types.union(count.keys())
    
    avg_counts = {cell_type:{} for cell_type in cell_types}
    std_counts = {cell_type:{} for cell_type in cell_types}
    states_by_celltypes = {cell_type: set() for cell_type in cell_types}
    
    for cell_type in cell_types:
        state_counts = {}    
        for states_pop in [count[cell_type] for count in counts if cell_type in count]:
            for state, pop in states_pop.items():
                if state not in state_counts.keys():
                    state_counts[state] = [pop]
                else:
                    state_counts[state].append(pop)
                    
        states_by_celltypes[cell_type] = states_by_celltypes[cell_type].union(state_counts.keys())
        
        for state, s_counts in state_counts.items():
            avg_counts[cell_type][state] = np.mean(s_counts)
            std_counts[cell_type][state] = np.std(s_counts)
            
    return (avg_counts, std_counts, states_by_celltypes)

nb_timesteps = np.zeros((args.replicates)).astype(np.int32)
for i in range(args.replicates):
    # print(nb_timesteps[i])
    if len(args.folder) > 0:
        path = os.path.join(args.folder, 'output_R%02d' % i)
    else:
        path = 'output_R%02d' % i

    while os.path.exists(os.path.join(path, 'output{:08d}.xml'.format(nb_timesteps[i]))):
        nb_timesteps[i] += 1

max_timesteps = np.max(nb_timesteps)
replicates = np.where(nb_timesteps == max_timesteps)[0].tolist()


with Pool(args.cores) as pool:
    res = pool.starmap(get_timestep, [(ts, replicates) for ts in range(max_timesteps)])


cell_types = set()
for avgs, _, _ in res:
    cell_types = cell_types.union(avgs.keys())

states_by_celltypes = {celltype: set() for celltype in cell_types}

for (avgs, stds, states) in res:
    for cell_type, avg in avgs.items():
        states_by_celltypes[cell_type] = states_by_celltypes[cell_type].union(avg.keys())

avgs_counts = {cell_type: { state: np.zeros(len(res)) for state in states_by_celltypes[cell_type]} for cell_type in cell_types}
stds_counts = {cell_type: { state: np.zeros(len(res)) for state in states_by_celltypes[cell_type]} for cell_type in cell_types}

for i, (avgs, stds, states) in enumerate(res):
    for cell_type in cell_types:        
        if cell_type in avgs.keys():
            for state, avg in avgs[cell_type].items():
                avgs_counts[cell_type][state][i] = avg
                stds_counts[cell_type][state][i] = stds[cell_type][state]
                

with open("trajs_states.pickle", "bw") as f_trajs:
    pickle.dump(avgs_counts, f_trajs)

with open("errors_states.pickle", "bw") as f_trajs:
    pickle.dump(stds_counts, f_trajs)
