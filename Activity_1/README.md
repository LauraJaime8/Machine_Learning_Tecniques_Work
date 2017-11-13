# Activity 1. Principal Component Analysis

This practice has helped us to achieve the basic knowledge related to extract data from a certain file, compute them and obtain the results in a way that makes possible the visual relationship between different features using Principal Component Analysis (PCA) or heatmaps.

## 1.- Load the data assigned
We have been assigned records of San Juan from 1997 to 2003, so such records have been loaded into a variable named "records". Some of them has no data so, unfilled fields have been replaced by 0.
Also there are features that only work as identificators (city, year, week of the wear and week start date) and, thus, have been removed.

## 2.- Extract the correlation among features and obtain conclusions.
Once the script is executed, it can be observed that there is a strong correlation between features 5 to 10 and 11. This makes sense because they are related to the total precipitation, 
mean air temperature, dew point and humidity level.

Also there is a correlation between features 4 and 12 (dirunal temperature and maximum air temperature).

Another correlations are shown: between features 0 and 1 (maximum and minimum temperature) and between features 2 and 3 (average temperature and total precipitation).

## 3.- Execute PCA and plot the results.
When plotting, some outliers are discovered: features 52, 104 and to a lesser extent, 358. This happens because records 52 and 104 have been fulfilled with 0 as no data was provided in some features.
