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




features = pd.read_csv('../Data/dengue_features_train.csv')
labels = pd.read_csv('../Data/dengue_labels_train.csv')
datafr = pd.merge(features, labels, on=['city', 'year', 'weekofyear'])

cities = ["sj", "iq"]

for i in cities:
    pca_plotting (i, datafr)
    correlation_plotting (i, datafr)


