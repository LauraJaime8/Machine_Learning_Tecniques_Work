#!/usr/bin/env python
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy

#http://docs.scipy.org/doc/scipy/reference/cluster.html
from scipy import cluster
from sklearn import preprocessing 
import sklearn.neighbors


# 0. Load Data
import loaddata
records, names = loaddata.load_data("dengue_features_train.csv")
 
#1. Normalization of the data
#http://scikit-learn.org/stable/modules/preprocessing.html
from sklearn import preprocessing 
min_max_scaler = preprocessing.MinMaxScaler()
records = min_max_scaler.fit_transform(records)
	
# 2. Compute the similarity matrix
dist = sklearn.neighbors.DistanceMetric.get_metric('chebyshev')
matsim = dist.pairwise(records)
avSim = numpy.average(matsim)
print "%s\t%6.2f" % ('Average Distance', avSim)

# 3. Building the Dendrogram	
# http://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.linkage.html#scipy.cluster.hierarchy.linkage
clusters = cluster.hierarchy.linkage(matsim, method = 'complete')
# http://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.cluster.hierarchy.dendrogram.html
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

