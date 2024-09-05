#!/usr/bin/env python
# coding: utf-8

# In[1]:

import scipy.io as sio
import numpy as np
import os
import sys
import argparse


parser = argparse.ArgumentParser(description='Process input')
parser.add_argument('--folder', type=str, default="", help='Choose which results to analyse')
parser.add_argument('--replicates', type=int, default=12, help='Inform how many replicated where done')

args = parser.parse_args()
os.chdir('../../')
sys.path.append('.')
from pyMCDS import pyMCDS

# In[ ]:

data = []
datatrace = []
    
temp1_DM = []  # 3
temp1_TC = []  # 3
temp1_TH1 = []  # 3
temp1_TH2 = []  # 3
temp1_BC = []  # 3
temp1_PS = []  # 3
temp1_DL = []  # 3


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



for j in replicates: 
    
        file_name = 'dm_tc.dat'
        if len(args.folder) > 0:
            path = os.path.join(args.folder, 'output_R'+str("%02d"%j))
        else:
            path = 'output_R'+str("%02d"%j)
        
        # os.chdir(path)
        d = np.loadtxt(os.path.join(path, file_name))
        # os.chdir('../')

        DM = d[:,0]
        TC = d[:,1]
        TH1 = d[:,2]
        TH2 = d[:,3]
        BC = d[:,6]
        PS = d[:,7]
        DL = d[:,12]
            
        temp1_DM.append( DM )
        temp1_TC.append( TC )
        temp1_TH1.append( TH1 )
        temp1_TH2.append( TH2 )
        temp1_BC.append( BC )
        temp1_PS.append( PS )
        temp1_DL.append( DL )

aDM = np.asarray(temp1_DM)
aTC = np.asarray(temp1_TC)
aTH1 = np.asarray(temp1_TH1)
aTH2 = np.asarray(temp1_TH2)
aBC = np.asarray(temp1_BC)
aPS = np.asarray(temp1_PS)
aDL = np.asarray(temp1_DL)

meanDM = np.mean(aDM, axis=0)
meanTC = np.mean(aTC, axis=0)
meanTH1 = np.mean(aTH1, axis=0)
meanTH2 = np.mean(aTH2, axis=0)
meanBC = np.mean(aBC, axis=0)
meanPS = np.mean(aPS, axis=0)
meanDL = np.mean(aDL, axis=0)

stdDM = np.std(aDM, axis=0)
stdTC = np.std(aTC, axis=0)
stdTH1 = np.std(aTH1, axis=0)
stdTH2 = np.std(aTH2, axis=0)
stdBC = np.std(aBC, axis=0)
stdPS = np.std(aPS, axis=0)
stdDL = np.std(aDL, axis=0)

data.append( np.vstack((meanDM, meanTC, meanTH1, meanTH2, meanBC, meanPS, meanDL, stdDM, stdTC, stdTH1, stdTH2, stdBC, stdPS, stdDL)) )
datatrace= np.dstack([aDM,aTC,aTH1,aTH2,aBC,aPS,aDL])

timedata = np.asarray(data)
if len(args.folder) > 0:
    sio.savemat(os.path.join(args.folder, 'timeReplicateDat.mat'), {'timedata':timedata})
    sio.savemat(os.path.join(args.folder, 'timeTracesDat.mat'), {'tracedata':datatrace})

else:
    sio.savemat('timeReplicateDat.mat', {'timedata':timedata})
    sio.savemat('timeTracesDat.mat', {'tracedata':datatrace})


