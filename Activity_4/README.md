# Activity 4. Decision Trees for Regression

## 1.- Load the data assigned
In this activity, instead of using the loaddata.py file, pandas library has been used to read the different files required. They will be stored in a dataframe.

## 2.- Extract the correlation between features and total cases.
The correlation between features and total cases is computed in order to see which ones has higher values. This will help at the time of choosing the ones for the model construction.
Low correlation means there's no linear relationship; it doesn't mean there's no information in the feature that predicts the target.

<p align="center">
  <img src="https://github.com/RuthRML/Machine_Learning_Tecniques_Work/blob/master/Activity_4/Images/1.png">
</p>

## 3.- Feature Selection.
This step can be done once the information about correlation has been computed, looking at the resulting graph. Density plots of each feature are also drawn.

<p align="center">
  <img src="https://github.com/RuthRML/Machine_Learning_Tecniques_Work/blob/master/Activity_4/Images/2.png">
</p>

<p align="center">
  <img src="https://github.com/RuthRML/Machine_Learning_Tecniques_Work/blob/master/Activity_4/Images/3.png">
</p>

<p align="center">
  <img src="https://github.com/RuthRML/Machine_Learning_Tecniques_Work/blob/master/Activity_4/Images/4.png">
</p>


## 4.- Build a Decision Tree Model using your data.
The relevances of each feature are extracted. Also a cross validation analysis is computed to extract the maximum tree depth in which the cross validation score (cv score)
reaches the lowest value.

<p align="center">
  <img src="https://github.com/RuthRML/Machine_Learning_Tecniques_Work/blob/master/Activity_4/Images/5.png">
</p>
