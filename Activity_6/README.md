# Activity 6. Predictive Model Building.
This script has been modified to make the computations twice using a for-loop to make them for each city.

## 1.- Load the data.
For this activity, the whole data set has been used (both San Juan and Iquitos cities).

## 2.- Execute PCA Plotting.
Outliers from both cities has been deleted by doing a normalization of the data and the plotting of the PCA.

## 3.- Study the correlation.
Looking at the resulting correlation graphs, a set of features from each city can be extracted according to the scores obtained.

## 4.- Cross Validation.
In this method, the relevances of the features selected of each city are computed. Also a graph is shown in order to see the 
lowest correlation value.

## 5.- kNN Function.
For this task, k Nearest Neighbors algorithm (kNN) has been used to predict the total cases for each week of further years.
The results are stored in the .csv submission format for the competition. 