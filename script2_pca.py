# -*- coding: utf-8 -*-
"""
Created on Mon Nov 09 17:35:44 2015

@author: FranciscoP.Romero, Ruth, Diego y Laura
"""
import codecs
import matplotlib.pyplot as plt
import numpy
from sklearn.decomposition import PCA
from sklearn import preprocessing 

# 0. Load Data
f = codecs.open("dengue_features_train.csv", "r", "utf-8")
records = []
years = ["1997", "1998", "1999", "2000", "2001", "2002", "2003"]

for line in f:
    #Replace no-data fields
    while ",," in line:
        line = line.replace(",,", ",0,")
    
    #Replace last unfilled field
    while ",\n" in line:
       line = line.replace(",\n", ",0\n")
       
    row = line.split(",")
    
    if row[0] == "sj" and row[1] in years:
        for i in range (4):
            row.pop(0)
        
        if row != []:
            records.append(map(float, row))
            


#1. Normalization of the data
min_max_scaler = preprocessing.MinMaxScaler()
records = min_max_scaler.fit_transform(records)
       
#2. PCA Estimation
estimator = PCA (n_components = 2)
X_pca = estimator.fit_transform(records)

print(estimator.explained_variance_ratio_) 

#3.  plot 
numbers = numpy.arange(len(X_pca))
fig, ax = plt.subplots()
for i in range(len(X_pca)):
    plt.text(X_pca[i][0], X_pca[i][1], numbers[i]) 
plt.xlim(-1.5, 2.5)
plt.ylim(-1, 3.5)
ax.grid(True)
fig.tight_layout()
plt.show()

