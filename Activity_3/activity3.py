# -*- coding: utf-8 -*-

"""

Perform the K-means technique in the data of San Juan from 1997 to 2003.

@author Diego Andérica Richard, Ruth Rodríguez-Manzaneque López, Laura Jaime Villamayor

"""

import loaddata
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn import metrics

def plotdata(data,labels,name): #Def function plotdata
    fig, ax = plt.subplots()
    plt.scatter([row[0] for row in data], [row[1] for row in data], c = labels)
    ax.grid(True)
    fig.tight_layout()
    plt.title(name)
    plt.show()

# 0. Load Data.
records, names = loaddata.load_data("../Data/dengue_features_train.csv")

# 1. Normalization of the data.
min_max_scaler = preprocessing.MinMaxScaler()
records = min_max_scaler.fit_transform(records)

# 2. PCA Estimation.
estimator = PCA (n_components = 2)
X_pca = estimator.fit_transform(records)

print(estimator.explained_variance_ratio_)

# 3. Plotting the PCA Estimation.
labels = [0 for x in range(len(X_pca))]
plotdata(X_pca, labels, 'Initial')

# 4. Setting parameters (ad-hoc)
# 4.1. Defining the Parameters
init = 'k-means++' # Initialization method
iterations = 10 # to run 10 times with different random centroids to choose the final model as the one with the lowest SSE
max_iter = 300 # maximum number of iterations for each single run
tol = 1e-04 # controls the tolerance with regard to the changes in the within-cluster sum-squared-error to declare convergence
random_state = 0 # random

silhouettes = []

for i in range(2, 14):
    km = KMeans(i, init, n_init = iterations ,max_iter= max_iter, tol = tol,random_state = random_state)
    labels = km.fit_predict(X_pca)
    silhouettes.append(metrics.silhouette_score(X_pca, labels))

# 4.2. Plot Silhouette technique to see which value for K is better.
plt.plot(range(2,14), silhouettes , marker='o')
plt.xlabel('Number of clusters')
plt.ylabel('Silhouette')
plt.show()

print "Due to the fact that 2 has the higher silhouette, this value will be used to set the value of k\n"

# 5. Setting k value
k = 2

# 6. Clustering execution (K-Means)
km = KMeans(k, init, n_init = iterations, max_iter= max_iter, tol = tol,random_state = random_state)
labels = km.fit_predict(X_pca)

# 7. Plotting final results
plotdata(X_pca, labels, init)
