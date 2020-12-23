#%%
import os
import pandas as pd
from sklearn.preprocessing import FunctionTransformer
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import HistGradientBoostingRegressor
import numpy as np
from math import pi, sin, cos
import xgboost as XGB
from sklearn.ensemble import StackingRegressor

#import torch
import random
#from torch import nn
#import torch.optim as optim
#import torch.nn.functional as F
import copy

def fourrier_func(x):
    return 0.5 + 1/4*(1.0945*sin(2*pi/7*x) - 0.5697*cos(2*pi/7*x) + 
                      2.1256*sin(4*pi/7*x) + 0.3972*cos(4*pi/7*x) -
                      0.6531*sin(8*pi/7*x) + 1.1725*cos(8*pi/7*x))

def _encode_dates(X):
    # Encode the date information from the DateOfDeparture columns
    X.loc[:, 'year'] = X['DateOfDeparture'].dt.year
    X.loc[:, 'month'] = X['DateOfDeparture'].dt.month
    X.loc[:, 'day'] = X['DateOfDeparture'].dt.day
    X.loc[:, 'weekday'] = X['DateOfDeparture'].dt.weekday
    X.loc[:, 'week'] = X['DateOfDeparture'].dt.week
    X.loc[:, 'n_days'] = X['DateOfDeparture'].apply(
        lambda date: (date - pd.to_datetime("1970-01-01")).days
    )
    # Finally we can drop the original columns from the dataframe
    return X.drop(columns=["DateOfDeparture"])



def _merge_external_data_N(X):
    filepath = os.path.join(
        os.path.dirname(__file__), 'external_data.csv'
    )
    X = X.copy()  # to avoid raising SettingOnCopyWarning
    # Make sure that DateOfDeparture is of dtype datetime
    X.loc[:, "DateOfDeparture"] = pd.to_datetime(X['DateOfDeparture'])
    X = _encode_dates(X)
    data_nicolas = pd.read_csv(filepath)
    
    X.rename(columns={'Departure':'ori', 'Arrival':'dest'},
                  inplace=True)
    #data_nicolas = data_nicolas[['Departure', 'Arrival', 'year', 'month', 'day', 'distance km']]
    #data_nicolas.rename(columns={'ori':'Departure','dest':'Arrival'}, inplace=True)
    #print(data_nicolas.columns)
    data_nicolas.drop_duplicates(['year', 'month', 'day','ori', 'dest'], inplace=True)
    X_merged = X.merge(data_nicolas, on=['year', 'month', 'day','ori', 'dest'],
                             how='left')
    #X_merged = X_merged.dropna()
    #X_merged = X_merged[['Departure', 'Arrival', 'WeeksToDeparture', 'std_wtd', 'year', 'month', 'day', 'weekday', 'week', 'n_days', 'distance km']]
    #print(X_merged.isna().sum())
    #X_merged.to_csv('merged.csv')
    #X_merged.drop([16, 624], axis=0, inplace=True)
    #print(X_merged.isna().sum())
    #print(X_merged.shape)
    #X_merged.to_csv(os.path.join(
        #os.path.dirname(__file__),'merged.csv'))
    X_merged.drop(['seats','departures_scheduled','month_pas','year_pas','gdp_ori_1',
     'gdp_dest_1','tot_pas_mon_ori'], axis=1, inplace=True)
    X_merged['route'] = (X_merged['ori'].astype(str) + '-' + 
                         X_merged['dest'].astype(str))
    X_merged['pas/flight'] = (X_merged['passengers']/X_merged['departures_performed']/10)    
    X_merged['import_day'] = (X_merged['n_days']-15214).apply(fourrier_func)
    
    return X_merged



def get_estimator():
    data_merger_N = FunctionTransformer(_merge_external_data_N)

    #date_encoder = FunctionTransformer(_encode_dates)
    #date_cols = ["DateOfDeparture"]

    cat_columns = ["ori", "dest", 'is_holiday', 'route', 'year']
    date_columns = ['month', 'day', 'weekday', 'week', 'import_day']
    num_columns = ['WeeksToDeparture', 'std_wtd', 'distance',
                   'departures_performed','passengers','import_route',
                   'import_month', 'load', 'cancel_rate', 'fuel_price',
                   'ratio M/F_ori', 'ratio M/F_dest', 'gdp_ori_2',
                   'gdp_dest_2', 'pop_metro_ori', 'pop_metro_dest',
                   'work_pop_ratio_ori', 'work_pop_ratio_dest',
                   'gdp_per_capita_ori', 'gdp_per_capita_dest', 'pas/flight', 'n_days']
    cat_encoder = OrdinalEncoder(categories='auto')
    num_encoder = StandardScaler()
    preprocessor = make_column_transformer(
        (cat_encoder, cat_columns), 
        (num_encoder, num_columns),
        remainder='passthrough'
    )
    
    xgb_reg = XGB.XGBRegressor( 
        gamma=0.09,
        learning_rate=0.14, 
        max_depth= 7,
        min_child_regressor=18,
        n_estimators=1200,
        reg_alpha=1,
        reg_lambda=2,
        subsample=0.9
    )

    hist_reg = HistGradientBoostingRegressor(
        learning_rate=0.0701793626657323,
        max_iter=1200,
        max_leaf_nodes=15,
        max_depth=7,
        min_samples_leaf=17,
        l2_regularization=6.057510134290272,
        max_bins=248
    )
 
    regressor = StackingRegressor([('xbg', xgb_reg), ('hist', hist_reg)], verbose=1)



    #regressor = RandomForestRegressor(
        #n_estimators=10, max_depth=10, max_features=10, n_jobs=4
    #)
    #log_clf = LogisticRegression()
    #rnd_clf = RandomForestRegressor()
    #svm_clf = SVC(probability=True)
    #xgb_clf = xgb.XGBRegressor()
    #adb_clf = AdaBoostRegressor()
    #grad_clf = GradientBoostingRegressor()
    #regressor = StackingRegressor(estimators=[('xgb', xgb_clf), ('grad', grad_clf), ('adb', adb_clf), ('rndf', rnd_clf)])

    return make_pipeline(data_merger_N, preprocessor, regressor)
# %%
