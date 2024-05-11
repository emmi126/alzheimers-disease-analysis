# file to run hierarchical clustering for one subject example

# Suppress deprecation warning
from numba.core.errors import NumbaDeprecationWarning, NumbaPendingDeprecationWarning
import warnings

warnings.simplefilter('ignore', category=NumbaDeprecationWarning)
warnings.simplefilter('ignore', category=NumbaPendingDeprecationWarning)

import matplotlib.pyplot as plt


import unit_utils

import subjects
import umap
import umap.plot

subs = ['25-10', '18-1'] # ['25-10', '36-1', '36-3', '64-30', '59-2', '62-1']

for sub in subs:

    subject = subjects.subjects[sub]

    n_neighbors = 100

    data = subjects.load_data(subject, units=True)

    pyr_units = data['olm']['pyr_units']

    binsize = 100
    ts = data['olm']['Sleep2']['ts']

    ts_limited = ts[0:2000*60*60]
    z_matrix1, bins = unit_utils.z_spike_matrix(ts_limited, pyr_units, binsize)

    reducer1 = umap.UMAP(n_neighbors=n_neighbors)

    embedding1 = reducer1.fit_transform(z_matrix1.T)
    embedding1.shape
    plt.scatter(
        embedding1[0:1000, 0],
        embedding1[0:1000, 1], marker='.')