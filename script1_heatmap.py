# -*- coding: utf-8 -*-
"""
Created on Tue Nov 10 17:38:16 2015

@author: FranciscoP.Romero, Ruth, Diego y Laura
"""

import codecs
from numpy import corrcoef, transpose, arange
from pylab import pcolor, show, colorbar, xticks, yticks
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt

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




# plotting the correlation matrix
#http://glowingpython.blogspot.com.es/2012/10/visualizing-correlation-matrices.html
R = corrcoef(transpose(records))
pcolor(R)
colorbar()
yticks(arange(0,21),range(0,21))
xticks(arange(0,21),range(0,21))
show()


# http://stanford.edu/~mwaskom/software/seaborn/examples/many_pairwise_correlations.html
# Generate a mask for the upper triangle
sns.set(style="white")
mask = np.zeros_like(R, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(11, 9))

# Generate a custom diverging colormap
cmap = sns.diverging_palette(200, 10, as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(R, mask=mask, cmap=cmap, vmax=.8,
            square=True, xticklabels=2, yticklabels=2,
            linewidths=.5, cbar_kws={"shrink": .5}, ax=ax)