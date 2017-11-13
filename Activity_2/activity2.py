#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""

@author: Ruth Rodríguez-Manzaneque López, Diego Andérica Richard y Laura Jaime Villamayor

"""

import matplotlib.pyplot as plt
import numpy
from scipy import cluster
from sklearn import preprocessing 
import sklearn.neighbors

import loaddata

# 0. Load Data
records, names = loaddata.load_data("../Data/dengue_features_train.csv")
 
# 1. Normalization of the data
min_max_scaler = preprocessing.MinMaxScaler()
records = min_max_scaler.fit_transform(records)
	
# 2. Compute the similarity matrix
dist = sklearn.neighbors.DistanceMetric.get_metric('chebyshev')
matsim = dist.pairwise(records)
avSim = numpy.average(matsim)
print "%s\t%6.2f" % ('Average Distance', avSim)

# 3. Building the Dendrogram	
clusters = cluster.hierarchy.linkage(matsim, method = 'complete')
cut_level = 6
cluster.hierarchy.dendrogram(clusters, color_threshold = cut_level)
plt.show()

# 4. Characterization
labels = cluster.hierarchy.fcluster(clusters, cut_level, criterion = 'distance')
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
print ('Estimated number of clusters: %d' % n_clusters_)

for c in range(1, n_clusters_ + 1):
    print 'Group', c
    
    for i in range(len(records[0])):
        column = [row[i] for j,row in enumerate(records) if labels[j] == c]
        if len(column) != 0:
            print i, numpy.mean(column)
