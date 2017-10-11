import matplotlib.pyplot as plt
import numpy


#http://docs.scipy.org/doc/scipy/reference/cluster.html
from scipy import cluster
from sklearn import preprocessing 
import sklearn.neighbors

# 0. Load Data
import loaddata
records,names = loaddata.load_data("dengue_features_train.csv")
#Transposicion de matriz. Elementos a características.
features = numpy.transpose(records);

#1. Normalization of the data
#http://scikit-learn.org/stable/modules/preprocessing.html
from sklearn import preprocessing 
min_max_scaler = preprocessing.MinMaxScaler()
features_norm = min_max_scaler.fit_transform(features)
	
# 2. Compute the similarity matrix
dist = sklearn.neighbors.DistanceMetric.get_metric('euclidean')
matsim = dist.pairwise(features_norm)
avSim = numpy.average(matsim)
print "%s\t%6.2f" % ('Average Distance', avSim)

# 3. Building the Dendrogram	
# http://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.linkage.html#scipy.cluster.hierarchy.linkage
clusters = cluster.hierarchy.linkage(matsim, method = 'complete')
cut_level = 6
# http://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.cluster.hierarchy.dendrogram.html
cluster.hierarchy.dendrogram(clusters, color_threshold = cut_level, labels = names, leaf_rotation=90)
plt.show()