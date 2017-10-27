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
records,names = loaddata.load_data("../Data/dengue_features_train.csv")
features = numpy.transpose(records); # Matrix transponse: elements to features

# 1. Normalization of the data
min_max_scaler = preprocessing.MinMaxScaler()
features_norm = min_max_scaler.fit_transform(features)
	
# 2. Compute the similarity matrix
dist = sklearn.neighbors.DistanceMetric.get_metric('euclidean')
matsim = dist.pairwise(features_norm)
avSim = numpy.average(matsim)
print "%s\t%6.2f" % ('Average Distance', avSim)

# 3. Building the Dendrogram	
clusters = cluster.hierarchy.linkage(matsim, method = 'complete')
cut_level = 6
cluster.hierarchy.dendrogram(clusters, color_threshold = cut_level, 
                             labels = names, leaf_rotation = 90)
plt.show()
