import matplotlib.pyplot as plt

import unit_utils
import subjects
import numpy as np
import umap
import umap.plot
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage

# Suppress deprecation warning
from numba.core.errors import NumbaDeprecationWarning, NumbaPendingDeprecationWarning
import warnings

warnings.simplefilter('ignore', category=NumbaDeprecationWarning)
warnings.simplefilter('ignore', category=NumbaPendingDeprecationWarning)

n_neighbors = 100

subject = subjects.subjects["18-1"]
distance = 50

data = subjects.load_data(subject, units=True)
pyr_units = data['hab']['int_units']
binsize = 50
ts = data['olm']['Sleep1']['ts']

ts_limited = ts[0:2000 * 60 * 60]
z_matrix1, bins = unit_utils.z_spike_matrix(ts_limited, pyr_units, binsize)

reducer1 = umap.UMAP(n_neighbors=n_neighbors)
embedding1 = reducer1.fit_transform(z_matrix1.T)

# Reduce the number of clusters for hierarchical clustering
max_clusters = 100
if embedding1.shape[0] > max_clusters:
    clustering = AgglomerativeClustering(n_clusters=max_clusters)
else:
    clustering = AgglomerativeClustering(n_clusters=None, distance_threshold=distance)

y_predict = clustering.fit_predict(embedding1)

# Compute the linkage matrix
Z = linkage(embedding1, method='ward')

# Plot the dendrogram
plt.title("Hierarchical Clustering Dendrogram")
dendrogram(Z, truncate_mode="lastp", p=max_clusters if embedding1.shape[0] > max_clusters else clustering.n_clusters_)
plt.xlabel("Number of points in node (or index of point if no parenthesis).")
plt.show()

# Select a specific cluster based on its size
unique_clusters, cluster_counts = np.unique(y_predict, return_counts=True)
largest_cluster_idx = np.argmax(cluster_counts)

# Plot only the largest cluster in the UMAP space
selected_cluster = embedding1[y_predict == largest_cluster_idx]

plt.scatter(selected_cluster[:, 0], selected_cluster[:, 1], marker='.')
plt.xlabel('UMAP Dimension 1')
plt.ylabel('UMAP Dimension 2')
plt.show()