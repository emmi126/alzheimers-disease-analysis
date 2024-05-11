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

# Try PCA transformation

subject = subjects.subjects["18-1"]

data = subjects.load_data(subject, units=True)

int_units = data['hab']['int_units']

binsize = 10
ts = data['hab']['Sleep1']['ts']
z_matrix, bins = unit_utils.z_spike_matrix(ts, int_units, binsize)




ripple_windows = data['hab']['Sleep1']['ripple_windows']
r_bins, r_start_bins = unit_utils.ripple_bins(bins, ripple_windows, ts)


sns.heatmap(z_matrix)
plt.show()


# try sleep2

binsize = 10
ripple_windows = data['hab']['Sleep2']['ripple_windows']
ts2 = data['hab']['Sleep2']['ts']
z_matrix_2, bins_2 = unit_utils.z_spike_matrix(ts2, int_units, binsize)

# z_ripple = z_matrix[:, r_bins]

pca = PCA()

# concate the two for fit
pca.fit(np.concatenate((z_matrix.T, z_matrix_2.T)))
#z_transformed = pca.fit_transform(z_matrix.T)
z_transformed = pca.transform(z_matrix.T)

z_transformed2 = pca.transform(z_matrix_2.T)





# try isomaps
# iso = Isomap(n_neighbors=8)
# iso.fit(z_ripple.T)
# z_manifold = iso.transform(z_ripple.T)


#t = np.arange(z_manifold.shape[0])
#plt.scatter(z_manifold[:, 0], z_manifold[:, 1], c=t)
#z_manifold_all = iso.transform(z_matrix.T)



fig = plt.figure()
ax = fig.add_subplot(projection='3d')
#ax.scatter(z_transformed[:,0], z_transformed[:,1], z_transformed[:,2], marker='.', s=4)
ax.scatter(z_transformed[:, 0], z_transformed[:, 1], z_transformed[:, 2], marker='.', s=4)
ax.scatter(z_transformed2[:, 0], z_transformed2[:, 1], z_transformed2[:, 2], marker='.', s=4)
ax.set_xlabel('PC1')
ax.set_ylabel('PC2')
ax.set_zlabel('PC3')

plt.figure()
#plt.scatter(z_transformed[:,0], z_transformed[:,1],  marker='.', s=4)
#t= np.arange(0,z_ripple_transformed.shape[0])
plt.scatter(z_transformed[:, 0], z_transformed[:, 1], marker='.', s=4)
plt.scatter(z_transformed2[:, 0], z_transformed2[:, 1], marker='.', s=4)

plt.colorbar()
# try other pcs
plt.show()


plt.figure()
length = z_transformed.shape[0]
#plt.scatter(z_transformed[:,0], z_transformed[:,1],  marker='.', s=4)
#t= np.arange(0,z_ripple_transformed.shape[0])
plt.scatter(z_transformed[:, 0], z_transformed[:, 1], marker='.', s=4, c="Blue")
#plt.figure()
plt.scatter(z_transformed2[:, 0], z_transformed2[:, 1], marker='.', s=4, c="Red")
plt.scatter(z_transformed2[:length, 0], z_transformed2[:length, 1], marker='.', s=4, c='Orange')
# try other pcs
plt.show()