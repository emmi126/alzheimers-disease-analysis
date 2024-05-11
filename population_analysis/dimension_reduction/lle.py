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
from sklearn.manifold import Isomap, LocallyLinearEmbedding

import subjects


# Try LLE

subject = subjects.subjects['18-1']

data = subjects.load_data(subject, units=True)

pyr_units = data['olm']['pyr_units']

# Define time intervals
sleep1_start_time = np.amin(data['olm']['Sleep1']['ts'])
sleep1_end_time = sleep1_start_time + 10 * 60 * 1000000

sleep2_start_time = np.amin(data['olm']['Sleep2']['ts'])
sleep2_end_time = sleep2_start_time + 10 * 60 * 1000000

# Generate spike matrices for Sleep 1 and Sleep 2
binsize = 100

ts_sleep1 = data['olm']['Sleep1']['ts'][(data['olm']['Sleep1']['ts'] >= sleep1_start_time) & (data['olm']['Sleep1']['ts'] < sleep1_end_time)]
ts_sleep2 = data['olm']['Sleep2']['ts'][(data['olm']['Sleep2']['ts'] >= sleep2_start_time) & (data['olm']['Sleep2']['ts'] < sleep2_end_time)]

z_matrix_sleep1, bins_1 = unit_utils.z_spike_matrix(ts_sleep1, pyr_units, binsize)
sns.heatmap(z_matrix_sleep1)
plt.show()

z_matrix_sleep2, bins_2 = unit_utils.z_spike_matrix(ts_sleep2, pyr_units, binsize)
sns.heatmap(z_matrix_sleep2)
plt.show()

# Perform LLE transformation
n_neighbors = 15
lle = LocallyLinearEmbedding(n_neighbors=n_neighbors)
lle.fit(z_matrix_sleep1.T)
z_manifold_sleep1 = lle.transform(z_matrix_sleep1.T)
z_manifold_sleep2 = lle.transform(z_matrix_sleep2.T)

# Plot the results
length = z_manifold_sleep1.shape[0]

plt.figure()
plt.scatter(z_manifold_sleep1[:, 0], z_manifold_sleep1[:, 1], marker='.', s=4, c='Blue')
plt.scatter(z_manifold_sleep2[:, 0], z_manifold_sleep2[:, 1], marker='.', s=4, c='Red')
plt.scatter(z_manifold_sleep2[0:length, 0], z_manifold_sleep2[0:length, 1], marker='.', s=4, c='Orange')
plt.show()
