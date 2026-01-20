# -*- coding: utf-8 -*-
"""
Created on Tue Jun 29 16:35:39 2021

@author: gennady
"""

import sys
import numpy as np
import pandas as pd
from sklearn import manifold

fname='distances4tsne-1'
if len(sys.argv)>1 and sys.argv[1] is not None:
    fname=sys.argv[1]
A = np.genfromtxt(fname+'.csv',delimiter=',',dtype=None)

perplexity = 30
if len(sys.argv)>2 and sys.argv[2] is not None:
    perplexity=int(sys.argv[2])

model = manifold.TSNE(metric="precomputed", init="random", perplexity=perplexity)
Y = model.fit_transform(A)
tsne_df = pd.DataFrame({'X':Y[:,0],'Y':Y[:,1]})
tsne_df.to_csv(fname+'_out_p'+str(perplexity)+'.csv')

