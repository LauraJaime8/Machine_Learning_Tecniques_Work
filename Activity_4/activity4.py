# -*- coding: utf-8 -*-
"""

@author Diego Andérica Richard, Ruth Rodríguez-Manzaneque López, Laura Jaime Villamayor

"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats.stats import pearsonr
from sklearn.tree import DecisionTreeRegressor
from sklearn.cross_validation import cross_val_score
from tabulate import tabulate

# 0. Load Data
city = "sj"
years = ["1997", "1998", "1999", "2000", "2001", "2002", "2003"]

features = pd.read_csv('../Data/dengue_features_train.csv')
labels = pd.read_csv('../Data/dengue_labels_train.csv')
datafr = pd.merge(features, labels, on=['city', 'year', 'weekofyear'])

years_chosen = datafr['year'].isin(years)
city = datafr['city'] == city

datafr = datafr[years_chosen & city]
datafr = datafr.fillna(0)

#1. Correlation between features and total cases
corr = [pearsonr(datafr['ndvi_ne'], datafr['total_cases'])[0],
        pearsonr(datafr['ndvi_nw'], datafr['total_cases'])[0],
        pearsonr(datafr['ndvi_se'], datafr['total_cases'])[0],
        pearsonr(datafr['ndvi_sw'], datafr['total_cases'])[0],
        pearsonr(datafr['precipitation_amt_mm'], datafr['total_cases'])[0],
        pearsonr(datafr['reanalysis_air_temp_k'], datafr['total_cases'])[0],
        pearsonr(datafr['reanalysis_avg_temp_k'], datafr['total_cases'])[0],
        pearsonr(datafr['reanalysis_dew_point_temp_k'], datafr['total_cases'])[0],
        pearsonr(datafr['reanalysis_max_air_temp_k'], datafr['total_cases'])[0],
        pearsonr(datafr['reanalysis_min_air_temp_k'], datafr['total_cases'])[0],
        pearsonr(datafr['reanalysis_precip_amt_kg_per_m2'], datafr['total_cases'])[0],
        pearsonr(datafr['reanalysis_relative_humidity_percent'], datafr['total_cases'])[0],
        pearsonr(datafr['reanalysis_sat_precip_amt_mm'], datafr['total_cases'])[0],
        pearsonr(datafr['reanalysis_specific_humidity_g_per_kg'], datafr['total_cases'])[0],
        pearsonr(datafr['reanalysis_tdtr_k'], datafr['total_cases'])[0],
        pearsonr(datafr['station_avg_temp_c'], datafr['total_cases'])[0],
        pearsonr(datafr['station_diur_temp_rng_c'], datafr['total_cases'])[0],
        pearsonr(datafr['station_max_temp_c'], datafr['total_cases'])[0],
        pearsonr(datafr['station_min_temp_c'], datafr['total_cases'])[0],
        pearsonr(datafr['station_precip_mm'], datafr['total_cases'])[0]]

features = ('ndvi_ne', 'ndvi_nw', 'ndvi_se', 'ndvi_sw', 'precipitation_amt_mm',
            'reanalysis_air_temp_k', 'reanalysis_avg_temp_k',
            'reanalysis_dew_point_temp_k', 'reanalysis_max_air_temp_k',
            'reanalysis_min_air_temp_k', 'reanalysis_precip_amt_kg_per_m2',
            'reanalysis_relative_humidity_percent', 'reanalysis_sat_precip_amt_mm',
            'reanalysis_specific_humidity_g_per_kg', 'reanalysis_tdtr_k',
            'station_avg_temp_c', 'station_diur_temp_rng_c', 'station_max_temp_c',
            'station_min_temp_c', 'station_precip_mm')

y_pos = np.arange(len(features))

plt.bar(y_pos, corr, align = 'center', alpha = 0.5)
plt.xticks(y_pos, features, rotation = 90)
plt.ylabel('Correlation')
plt.title('Correlation Features vs Total Cases')

plt.show()

# 2. Density Plots
datafr.plot(kind = 'density', subplots = True, layout = (6,4), sharex = False)
plt.show()

# 3. Decision Tree Model

# 3.1 Model Parametrization 
# Criterion: mse mean squared error, which is equal to variance reduction as feature selection criterion
# Splitter: best/random
# max_depth: low value avoids overfitting
regressor = DecisionTreeRegressor(criterion = 'mse', max_depth = 2, random_state = 0)

# 3.2 Model construction
features_selected = ['year', 'weekofyear', 'reanalysis_precip_amt_kg_per_m2', 'reanalysis_specific_humidity_g_per_kg', 'reanalysis_tdtr_k']
datafr_features = datafr[features_selected]
total_cases = datafr['total_cases']

regressor.fit(datafr_features, total_cases)

# 4. Cross Validation Analysis
# 4.1 Feature Relevances
list1 = zip(features_selected, regressor.feature_importances_)
print tabulate(list1, headers = ['Feature', 'Relevance'])

total_scores = []
for i in range(2, 30):
    regressor = DecisionTreeRegressor(max_depth = i)
    regressor.fit(datafr_features, total_cases)
    scores = -cross_val_score(regressor, datafr_features, total_cases,
                              scoring = 'neg_mean_absolute_error', cv = 10)
    total_scores.append(scores.mean())

plt.plot(range(2,30), total_scores, marker = 'o')
plt.xlabel('max_depth')
plt.ylabel('cv score')
plt.show()





