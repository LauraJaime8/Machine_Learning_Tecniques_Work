# -*- coding: utf-8 -*-

"""

@author: Ruth Rodríguez-Manzaneque López, Diego Andérica Richard y Laura Jaime Villamayor

"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn import preprocessing
from scipy.stats.stats import pearsonr
from sklearn.tree import DecisionTreeRegressor
from sklearn.cross_validation import cross_val_score
from tabulate import tabulate
from sklearn import neighbors
import csv
from sklearn.ensemble import RandomForestRegressor
from sklearn.naive_bayes import GaussianNB

def pca_plotting (city_name, datafr):
    city = datafr['city'] == city_name
    datafr = datafr[city]
    datafr = datafr.fillna(0)
    
    del datafr['city']
    del datafr['year']
    del datafr['weekofyear']
    del datafr['week_start_date']
    
    if city_name is "sj":
        datafr.drop(datafr.index[399], inplace=True)
        datafr.drop(datafr.index[87], inplace=True)
        datafr.drop(datafr.index[710], inplace=True)
        datafr.drop(datafr.index[709], inplace=True)
        datafr.drop(datafr.index[229], inplace=True)
        datafr.drop(datafr.index[138], inplace=True)
        datafr.drop(datafr.index[778], inplace=True)
        datafr.drop(datafr.index[438], inplace=True)
        datafr.drop(datafr.index[746], inplace=True)
        datafr.drop(datafr.index[446], inplace=True)
        datafr.drop(datafr.index[754], inplace=True)
    elif city_name is 'iq':
        datafr.drop(datafr.index[182], inplace=True)
        datafr.drop(datafr.index[442], inplace=True)
        datafr.drop(datafr.index[494], inplace=True)
        datafr.drop(datafr.index[441], inplace=True)
        datafr.drop(datafr.index[292], inplace=True)
        datafr.drop(datafr.index[293], inplace=True)
        datafr.drop(datafr.index[430], inplace=True)
        datafr.drop(datafr.index[233], inplace=True)
        datafr.drop(datafr.index[488], inplace=True)
        datafr.drop(datafr.index[487], inplace=True)
        datafr.drop(datafr.index[271], inplace=True)
        
    #1. Normalization of the data
    min_max_scaler = preprocessing.MinMaxScaler()
    records = min_max_scaler.fit_transform(datafr)
           
    #2. PCA Estimation
    estimator = PCA (n_components = 2)
    X_pca = estimator.fit_transform(records)
    
    print(estimator.explained_variance_ratio_) 
    
    #3. Plot 
    numbers = np.arange(len(X_pca))
    fig, ax = plt.subplots()
    
    for i in range(len(X_pca)):
        plt.text(X_pca[i][0], X_pca[i][1], numbers[i])
        
    plt.xlim(-1.5, 3.5)
    plt.ylim(-1, 3.5)
    ax.grid(True)
    fig.tight_layout()
    if city_name is "sj":
        plt.title('PCA - San Juan')
    elif city_name is "iq":
        plt.title('PCA - Iquitos')
    plt.show()
    
def correlation_plotting (city_name, datafr):
    city = datafr['city'] == city_name

    datafr = datafr[city]
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
    
    if city_name is "sj":
        plt.title('Correlation Features vs Total Cases (San Juan)')
    elif city_name is "iq":
        plt.title('Correlation Features vs Total Cases (Iquitos)')
    
    plt.show()

def crossvalidation (city_name, datafr):
    city = datafr['city'] == city_name
    
    if city_name is 'sj':
        print 'San Juan Cross-Validation:'
    elif city_name is 'iq':
        print 'Iquitos Cross-Validation:'
    
    datafr = datafr[city]
    datafr = datafr.fillna(0)
    
    if city_name is 'sj':
        regressor = DecisionTreeRegressor(criterion = 'mse', max_depth = 5, random_state = 0)
    elif city_name is 'iq':
        regressor = DecisionTreeRegressor(criterion = 'mse', max_depth = 3, random_state = 0)
    
    if city_name is 'sj':
        features_selected = ['year', 'weekofyear', 'reanalysis_precip_amt_kg_per_m2', 'reanalysis_specific_humidity_g_per_kg', 'station_avg_temp_c', 'station_max_temp_c', 'station_min_temp_c']
    elif city_name is 'iq':
        features_selected = ['year', 'weekofyear', 'reanalysis_precip_amt_kg_per_m2', 'reanalysis_specific_humidity_g_per_kg', 'reanalysis_tdtr_k', 'precipitation_amt_mm', 'reanalysis_tdtr_k', 'station_diur_temp_rng_c', 'reanalysis_sat_precip_amt_mm']
    
    datafr_features = datafr[features_selected]
    total_cases = datafr['total_cases']
    
    regressor.fit(datafr_features, total_cases)
        
    list1 = zip(datafr_features, regressor.feature_importances_)
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
    
def crossvalidationRandomForest (city_name, datafr):
    city = datafr['city'] == city_name
    
    if city_name is 'sj':
        print 'San Juan Cross-Validation:'
    elif city_name is 'iq':
        print 'Iquitos Cross-Validation:'
    
    datafr = datafr[city]
    datafr = datafr.fillna(0)
    
    if city_name is 'sj':
        regressor = RandomForestRegressor(n_estimators= 4, max_depth = 5, criterion='mae', random_state=0)
    elif city_name is 'iq':
        regressor = RandomForestRegressor(n_estimators= 4, max_depth = 3, criterion='mae', random_state=0)
    
    if city_name is 'sj':
        features_selected = ['year', 'weekofyear', 'reanalysis_precip_amt_kg_per_m2', 'reanalysis_specific_humidity_g_per_kg', 'station_avg_temp_c', 'station_max_temp_c', 'station_min_temp_c']
    elif city_name is 'iq':
        features_selected = ['year', 'weekofyear', 'reanalysis_precip_amt_kg_per_m2', 'reanalysis_specific_humidity_g_per_kg', 'reanalysis_tdtr_k', 'precipitation_amt_mm', 'reanalysis_tdtr_k', 'station_diur_temp_rng_c', 'reanalysis_sat_precip_amt_mm']
    
    datafr_features = datafr[features_selected]
    total_cases = datafr['total_cases']
    
    regressor.fit(datafr_features, total_cases)
        
    list1 = zip(datafr_features, regressor.feature_importances_)
    print tabulate(list1, headers = ['Feature', 'Relevance'])

    total_scores = []
   
    for i in range(2, 30):
        regressor = RandomForestRegressor(max_depth = i)
        regressor.fit(datafr_features, total_cases)
        scores = -cross_val_score(regressor, datafr_features, total_cases,
                                  scoring = 'neg_mean_absolute_error', cv = 10)
        total_scores.append(scores.mean())
    
    plt.plot(range(2,30), total_scores, marker = 'o')
    plt.xlabel('max_depth')
    plt.ylabel('cv score')
    plt.show()
    
def randomForest (city_name, datafr, test, submission, list_csv):
    
    city = datafr['city'] == city_name
    city_t = test['city'] == city_name
    
    datafr = datafr[city]
    test = test[city_t]
    datafr = datafr.fillna(0)
    test = test.fillna(0)
    
    if city_name is 'sj':
        X = ['year', 'weekofyear', 'reanalysis_precip_amt_kg_per_m2', 'reanalysis_specific_humidity_g_per_kg', 'station_avg_temp_c', 'station_max_temp_c', 'station_min_temp_c']
        X_test = ['year', 'weekofyear', 'reanalysis_precip_amt_kg_per_m2', 'reanalysis_specific_humidity_g_per_kg', 'station_avg_temp_c', 'station_max_temp_c', 'station_min_temp_c']
        list_csv.append(('city', 'year', 'weekofyear', 'total_cases'))
        rf = RandomForestRegressor(n_estimators= 4, max_depth = 5, criterion='mae', random_state=0)
    elif city_name is 'iq':
        X = ['year', 'weekofyear', 'reanalysis_precip_amt_kg_per_m2', 'reanalysis_specific_humidity_g_per_kg', 'reanalysis_tdtr_k', 'precipitation_amt_mm', 'reanalysis_tdtr_k', 'station_diur_temp_rng_c', 'reanalysis_sat_precip_amt_mm']
        X_test = ['year', 'weekofyear', 'reanalysis_precip_amt_kg_per_m2', 'reanalysis_specific_humidity_g_per_kg', 'reanalysis_tdtr_k', 'precipitation_amt_mm', 'reanalysis_tdtr_k', 'station_diur_temp_rng_c', 'reanalysis_sat_precip_amt_mm']
        rf = RandomForestRegressor(n_estimators= 4, max_depth = 3, criterion='mae', random_state=0)

    X = datafr[X]
    y = datafr['total_cases'] 
    X_test = test[X_test]
        
    prediction_model = rf.fit(X, y)
    prediction = prediction_model.predict(X_test)

    cont = 0

    for i, row in submission.iterrows():
        
        if city_name == 'sj' and row.city == 'sj':
            list_csv.append((row.city, int(round(row.year)), int(round(row.weekofyear)), int(round(prediction[cont]))))
            cont += 1
        elif city_name == 'iq' and row.city == 'iq':
            list_csv.append((row.city, int(round(row.year)), int(round(row.weekofyear)), int(round(prediction[cont]))))
            cont += 1

    return list_csv
    
def knnfunction (city_name, datafr, test, submission, list_csv):
    
    city = datafr['city'] == city_name
    city_t = test['city'] == city_name
    
    datafr = datafr[city]
    test = test[city_t]
    datafr = datafr.fillna(0)
    test = test.fillna(0)
    
    if city_name is 'sj':
        X = ['year', 'weekofyear', 'reanalysis_precip_amt_kg_per_m2', 'reanalysis_specific_humidity_g_per_kg', 'station_avg_temp_c', 'station_max_temp_c', 'station_min_temp_c']
        X_test = ['year', 'weekofyear', 'reanalysis_precip_amt_kg_per_m2', 'reanalysis_specific_humidity_g_per_kg', 'station_avg_temp_c', 'station_max_temp_c', 'station_min_temp_c']
        list_csv.append(('city', 'year', 'weekofyear', 'total_cases'))
    elif city_name is 'iq':
        X = ['year', 'weekofyear', 'reanalysis_precip_amt_kg_per_m2', 'reanalysis_specific_humidity_g_per_kg', 'reanalysis_tdtr_k', 'precipitation_amt_mm', 'reanalysis_tdtr_k', 'station_diur_temp_rng_c', 'reanalysis_sat_precip_amt_mm']
        X_test = ['year', 'weekofyear', 'reanalysis_precip_amt_kg_per_m2', 'reanalysis_specific_humidity_g_per_kg', 'reanalysis_tdtr_k', 'precipitation_amt_mm', 'reanalysis_tdtr_k', 'station_diur_temp_rng_c', 'reanalysis_sat_precip_amt_mm']
        
    X = datafr[X]
    y = datafr['total_cases'] 
    X_test = test[X_test]
    
    n_neighbors = 5

    knn = neighbors.KNeighborsRegressor(n_neighbors, weights='uniform')
    prediction_model = knn.fit(X, y)
    prediction = prediction_model.predict(X_test)

    cont = 0

    for i, row in submission.iterrows():
        
        if city_name == 'sj' and row.city == 'sj':
            list_csv.append((row.city, int(round(row.year)), int(round(row.weekofyear)), int(round(prediction[cont]))))
            cont += 1
        elif city_name == 'iq' and row.city == 'iq':
            list_csv.append((row.city, int(round(row.year)), int(round(row.weekofyear)), int(round(prediction[cont]))))
            cont += 1

    return list_csv
    
    
if __name__ == '__main__':
    features = pd.read_csv('../Data/dengue_features_train.csv')
    labels = pd.read_csv('../Data/dengue_labels_train.csv')
    datafr = pd.merge(features, labels, on=['city', 'year', 'weekofyear'])
    test = pd.read_csv('../Data/dengue_features_test.csv')
    submission = pd.read_csv('../Data/submission_format.csv')
    cities = ["sj", "iq"]
    list_csv = []
    
    for i in cities:
        pca_plotting (i, datafr)
        correlation_plotting (i, datafr)
        crossvalidationRandomForest(i, datafr)
        list_csv = (randomForest(i, datafr, test, submission, list_csv))

    with open("../Data/output_activity6.csv", "wb") as f:
        writer = csv.writer(f)
        writer.writerows(list_csv)
        f.close()
        
        
