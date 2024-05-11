import matplotlib.pyplot as plt
import numpy as np
import pandas
import pandas as pd

import unit_utils
import os

import scipy.io as sio
from scipy import stats
import pickle

from sklearn.decomposition import FastICA
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import Isomap

import subjects

# Try Isomap transformation

subject = subjects.subjects["18-1"]

data = subjects.load_data(subject, units=True)

pyr_units = data['olm']['pyr_units']


''' SLEEP 1 '''
binsize = 50
ts_all = data['olm']['Sleep1']['ts']

# Time interval
start_time = np.amin(ts_all)
end_time = np.amin(ts_all) + 40 * 60 * 1000000 ###########################################

ts = ts_all[(ts_all >= start_time) & (ts_all < end_time)]
z_matrix, bins = unit_utils.z_spike_matrix(ts, pyr_units, binsize)

sns.heatmap(z_matrix)
plt.show()

''' SLEEP 2 '''
binsize = 50
ts2_all = data['olm']['Sleep2']['ts']

# Time interval
start_time = np.amin(ts2_all)
end_time = np.amin(ts2_all) + 40 * 60 * 1000000 ###########################################

ts2 = ts2_all[(ts2_all >= start_time) & (ts2_all < end_time)]
z_matrix_2, bins_2 = unit_utils.z_spike_matrix(ts2, pyr_units, binsize)

sns.heatmap(z_matrix_2)
plt.show()

#ripple_windows = data['olm']['Sleep1']['ripple_windows']
#r_bins, r_start_bins = unit_utils.ripple_bins(bins,ripple_windows,ts)
#r_matrix = unit_utils.ripple_spike_matrix(ts, ripple_windows, pyr_units)

#ripple_windows = data['olm']['Sleep2']['ripple_windows']
#ts2 = data['olm']['Sleep2']['ts']
#r_matrix_2 = unit_utils.ripple_spike_matrix(ts2, ripple_windows, pyr_units)


#z_ripple = z_matrix[:,r_bins]

#z_ripple_transformed = pca.transform(z_ripple.T)

# Isomaps

# iso = Isomap(n_neighbors=8) # 15 # 70
iso = Isomap(n_neighbors=150) ###############################################################################
iso.fit(z_matrix.T)
z_manifold = iso.transform(z_matrix.T)
z_manifold_2 = iso.transform(z_matrix_2.T)

#t = np.arange(z_manifold.shape[0])
#plt.scatter(z_manifold[:, 0], z_manifold[:, 1], c=t)
#z_manifold_all = iso.transform(z_matrix.T)

"""fig = plt.figure()
ax = fig.add_subplot(projection='3d')
#ax.scatter(z_transformed[:,0], z_transformed[:,1], z_transformed[:,2], marker='.', s=4)
ax.scatter(z_manifold[:,0], z_manifold[:,1], z_ripple_transformed[:,2], marker='.', s=4)
ax.set_xlabel('PC1')
ax.set_ylabel('PC2')
ax.set_zlabel('PC3')"""

length = z_manifold.shape[0]

plt.figure()
#t= np.arange(0,z_manifold.shape[0])

plt.scatter(z_manifold[:,0], z_manifold[:,1],  marker='.', s=4,c='Blue')
#plt.figure()

plt.scatter(z_manifold_2[:,0], z_manifold_2[:,1],  marker='.', s=4,c='Red')

#plt.show()
#plt.scatter(z_manifold[:,0], z_manifold[:,1],  marker='.', s=4, c=t)

plt.scatter(z_manifold_2[0:length,0], z_manifold_2[0:length:,1],  marker='.', s=4,c='Orange')

plt.show()