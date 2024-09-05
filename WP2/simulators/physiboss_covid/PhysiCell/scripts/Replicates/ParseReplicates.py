#!/usr/bin/env python
# coding: utf-8

import scipy.io as sio
import numpy as np
import os
import sys
import argparse
os.chdir('../../')
sys.path.append('.')
from pyMCDS import pyMCDS

parser = argparse.ArgumentParser(description='Process input')
parser.add_argument('--folder', type=str, default="", help='Choose which results to analyse')
parser.add_argument('--replicates', type=int, default=12, help='Inform how many replicated where done')
parser.add_argument('--cores', type=int, default=10, help='Number of cores to use for analysing results')

args = parser.parse_args()

def get_values(path, k):
    
    str_name = 'output{:08d}.xml'.format(k)
    mcds = pyMCDS(str_name, path)  # /case1/run3/output

    cycle = mcds.data['discrete_cells']['cycle_model']
    cycle = cycle.astype(int)
    phase = mcds.data['discrete_cells']['ability_to_phagocytose_infected_cell']
    phase = phase.astype(int)
    active = mcds.data['discrete_cells']['activated_immune_cell']
    active = active.astype(int)
    ex = mcds.data['discrete_cells']['M2_phase']
    ex = ex.astype(int)
    ex2 = mcds.data['discrete_cells']['total_volume']
    ex2 = ex.astype(int)
    cell_type = mcds.data['discrete_cells']['cell_type']
    cell_type = cell_type.astype(int)

    CD8 = np.where((cell_type ==3) & (cycle < 100))
    mac1 = np.where((cell_type ==4) & (cycle < 100) & (active == 1) & (ex == 0))
    mac2 = np.where((cell_type ==4) & (cycle < 100) & (ex == 1))
    mac3 = np.where((cell_type ==4) & (cycle < 100) & (active == 0))
    mac4 = np.where((cell_type ==4) & (cycle < 100) & (phase == 1))
    mac5 = np.where((cell_type ==4) & (cycle < 100) & (ex2 > 6500))
    neut = np.where((cell_type ==5) & (cycle < 100))
    
    DC = np.where((cell_type ==6) & (cycle < 100))
    CD4 = np.where((cell_type ==7) & (cycle < 100))
    fib = np.where((cell_type ==8) & (cycle < 100))
    virI = np.sum(mcds.data['discrete_cells']['assembled_virion'])
    virm = mcds.data['continuum_variables']['virion']
    virs = np.sum(virm['data'])*8000
    vir = np.add(virI,virs)
    IFNm = mcds.data['continuum_variables']['interferon 1']
    IFN = np.sum(IFNm['data'])*8000
    Igm = mcds.data['continuum_variables']['Ig']
    Ig = np.sum(Igm['data'])*8000
    pIm = mcds.data['continuum_variables']['pro-inflammatory cytokine']
    pI = np.sum(pIm['data'])*8000
    aIm = mcds.data['continuum_variables']['anti-inflammatory cytokine']
    aI = np.sum(aIm['data'])*8000
    colm = mcds.data['continuum_variables']['collagen']
    col = np.sum(colm['data'])*8000
    epi = np.where((cell_type ==1) & (cycle < 100))
    
    return np.array(
        [len(CD8[0]), len(mac1[0]), len(mac2[0]), len(mac3[0]), len(mac4[0]), len(mac5[0]), 
        len(neut[0]), len(DC[0]), len(CD4[0]), len(fib[0]), 
        vir, IFN, Ig, pI, aI, col, len(epi[0])])

def get_timestep(k, replicates):
    
    data_timestep = []
    for j in replicates: 

        if len(args.folder) > 0:
            path = os.path.join(args.folder, 'output_R'+str("%02d"%j))
        else:
            path = 'output_R'+str("%02d"%j)
        
        data_timestep.append(get_values(path, k))
    
    stack_timestep = np.stack(data_timestep)
    mean_timestep = np.mean(stack_timestep, axis=0)
    std_timestep = np.std(stack_timestep, axis=0)
    
    # return np.concatenate([mean_timestep, std_timestep])
    return [stack_timestep, np.concatenate([mean_timestep, std_timestep])]


from multiprocessing import Pool

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

if len(replicates) < args.replicates:
    print("Detected failed replicates : %s" % str(np.where(nb_timesteps < max_timesteps)[0].tolist()))
    
with Pool(args.cores) as pool:
    res = pool.starmap(get_timestep, [(ts, replicates) for ts in range(max_timesteps)])


stacked_data = np.vstack([re[1] for re in res])
timedata = np.asarray([stacked_data.transpose()])

f_res = [re[0] for re in res]
datatrace = np.transpose(np.dstack(f_res),axes=[0,2, 1])


if len(args.folder) > 0:
    sio.savemat(os.path.join(args.folder, 'timeReplicate.mat'), {'timedata':timedata})
    sio.savemat(os.path.join(args.folder, 'timeTraces.mat'), {'tracedata':datatrace})

else:    
    sio.savemat('timeReplicate.mat', {'timedata':timedata})
    sio.savemat('timeTraces.mat', {'tracedata':datatrace})


