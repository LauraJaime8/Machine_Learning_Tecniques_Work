# Activity 2. Hierarchical Clustering

## 1.- Load and normalize the data assigned
We have been assigned records of San Juan from 1997 to 2003, so such records have been loaded into a variable named "records". Some of them has no data so, unfilled fields have been replaced by 0.
Also there are features that only work as identificators (city, year, week of the wear and week start date) and, thus, have been removed.

In this activity the first row of the file have been kept as it contains the labels (names of each feature) necessary in further questions.

## 2.- Compute the similarity matrix. Execute the hierarchical clustering algorithm.
It has been tested some cluster-distances-measures like Euclidean, Manhattan and Cebyshev. The last one is the chosen to run the computations because it has the minimum average distance
between the points (0.35).

## 3.- Cut the dendrogram and characterize the obtained groups. Assign a label to each group.
Depending on the measure chosen and linkage (complete or average) different dendograms are obtained. At the end, the one resulting from Chebyshev with complete linkage has been chosen.
After that, it has been cut to level 6 as is the most adequate to show the clusters.


<p align="center">
  <img src="https://github.com/RuthRML/Machine_Learning_Tecniques_Work/blob/master/Activity_2/Images/1.png">
</p>

There are four different clusters (starting from right to left):
* Purple. Includes some weeks that have the maximum precipitations (fall months).
* Blue. Includes weeks that have more precipitations than the mean of other seasons (summer months).
* Red. Includes weeks with very poor levels of precipitations.
* Green. This is the biggest group and it includes the rest of the weeks when the levels of precipitations are normal.

## 4.- Execute the hierarchical clustering algorithm using feature as elements.
There are several clusters (from left to right):
* The first group corresponds to the reanalysis of air temperature (kelvin).
* The red one corresponding to diurnal temperature ranges and satellite vegetation.
* The turquoise corresponding to precipitations, humidity and temperature.
* A cluster with an unique feature (reanalysis_relative_humidity_percent).
* Last one corresponding to total precipitations.


<p align="center">
  <img src="https://github.com/RuthRML/Machine_Learning_Tecniques_Work/blob/master/Activity_2/Images/2.png">
</p>




