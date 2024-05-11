import matplotlib.pyplot as plt
import numpy as np
import unit_utils
import seaborn as sns
from sklearn.cluster import AgglomerativeClustering
import subjects
from scipy.cluster.hierarchy import dendrogram, linkage

from sklearn.decomposition import PCA

# Define the subject and distance
subject = subjects.subjects['18-1']
distance = 50

# Load the data and perform PCA
data = subjects.load_data(subject, units=True)
pyr_units = data['olm']['pyr_units']
binsize = 50
ts = data['olm']['Sleep1']['ts']

# Choose a subsample within the time range
ts_limited = ts[0:2000 * 60 * 60]

# Subsample data to reduce memory usage
max_samples = 5000
if ts_limited.size > max_samples:
    subsample_indices = np.random.choice(np.arange(ts_limited.size), size=max_samples, replace=False)
    ts_limited = ts_limited[subsample_indices]

z_matrix, bins = unit_utils.z_spike_matrix(ts_limited, pyr_units, binsize=binsize)

# Perform PCA transformation
pca = PCA()
z_transformed = pca.fit_transform(z_matrix.T)

# Reduce the number of clusters for hierarchical clustering
max_clusters = 100
if z_transformed.shape[0] > max_clusters:
    clustering = AgglomerativeClustering(n_clusters=max_clusters)
else:
    clustering = AgglomerativeClustering(n_clusters=None, distance_threshold=distance)

y_predict = clustering.fit_predict(z_transformed)

# Compute the linkage matrix
Z = linkage(z_transformed, method='ward')

# Plot the dendrogram
plt.title("Hierarchical Clustering Dendrogram")
dendrogram(Z, truncate_mode="lastp", p=max_clusters if z_transformed.shape[0] > max_clusters else clustering.n_clusters_)
plt.xlabel("Number of points in node (or index of point if no parenthesis).")
plt.show()

# Select a specific cluster based on its size
unique_clusters, cluster_counts = np.unique(y_predict, return_counts=True)
largest_cluster_idx = np.argmax(cluster_counts)

# Plot only the largest cluster
selected_cluster = z_transformed[y_predict == largest_cluster_idx]

plt.scatter(selected_cluster[:, 0], selected_cluster[:, 1], c='b', label='Cluster %d' % largest_cluster_idx)
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.legend()
plt.show()