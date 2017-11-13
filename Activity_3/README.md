# Activity 3. K-Means

## 1.- Load and normalize the data assigned
We have been assigned records of San Juan from 1997 to 2003, so such records have been loaded into a variable named "records". Some of them has no data so, unfilled fields have been replaced by 0.
Also there are features that only work as identificators (city, year, week of the wear and week start date) and, thus, have been removed.

The first row of the file have been kept as it contains the labels (names of each feature) necessary in further questions but in this activity they are not used.

## 2.- Remove outliers.
The original number of elements in the dataset was 364, but there was some outliers (52, 104, 358, 103, 356) so they have been removed before making the computations related to clustering.
Thus, the final number of elements is 359 without outliers.

## 3.- Find the best value for k.
One of the things to do before applying k-means is try to find the best value for k (number of clusters). In this task the concept of "silhouette" has been used. This concept indicates the best 
number of k when it is closer to 1 and provides a representation of how well each object lies within its cluster. When applying it to the dataset, k=2 obtains the closest result to 1, so this 
will be the value to make the computations.

## 4.- Apply K-Means.
When executing K-Means, it can be chosen different options of initialization. In the script, the "kmeans++" algorithm has been used. In conclusion, both clusters seems to represent the 
periods of the year in which the climate conditions are similar.
