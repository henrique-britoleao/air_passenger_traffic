########################################################################
#                        Prototyping file
########################################################################

########################################################################
#                        Setup
########################################################################

# Importing packages
#%%
import pandas as pd
from matplotlib import pyplot as plt
import os
import problem
import numpy as np
import seaborn as sns
from math import pi, sin, cos

#%%
#DataFrame visualization options
pd.set_option('display.max_columns', None)

########################################################################
#                        Data Importing and merging
########################################################################

#%%
#import data
data, pas = problem.get_train_data()

#%%
def encode_dates(df, colunm_name):
    '''
    Encodes the DateTime information of a column of a data frame
    into: year, month, day, weekday, week, and number of days since
    1970-01-01. Deletes the original column. 

    Args: - df: Data Frame
          - column_name
    Out:  - DataFrame with column encoded
    '''
    
    # Encode the date information from the colunm_name
    df.loc[:, 'year'] = df[colunm_name].dt.year
    df.loc[:, 'month'] = df[colunm_name].dt.month
    df.loc[:, 'day'] = df[colunm_name].dt.day
    df.loc[:, 'weekday'] = df[colunm_name].dt.weekday
    df.loc[:, 'week'] = df[colunm_name].dt.week
    df.loc[:, 'n_days'] = df[colunm_name].apply(
        lambda date: (date - pd.to_datetime("1970-01-01")).days
    )
    # Finally we can drop the original columns from the dataframe
    return df.drop(columns=["DateOfDeparture"])

# %%
#Encoding datetime data
data['DateOfDeparture'] = pd.to_datetime(data['DateOfDeparture'])
data_encod = encode_dates(data, 'DateOfDeparture')
data_encod.head()

# %%
#import additinal data
add_data = pd.read_csv(os.path.join('processed_add_data',
                                     'aggregated_data_cor.csv'))
add_data.drop_duplicates(['year', 'month', 'day', 'ori', 'dest'],
                         inplace=True) #drop dupplicate routes and days
# %%
#solve name discrep
data_encod.rename(columns={'Departure':'ori', 'Arrival':'dest'},
                  inplace=True)

#merge original and additinal data
full_data = data_encod.merge(add_data, 
                             on=['year', 'month', 'day','ori', 'dest'],
                             how='left')

#%%
########################################################################
#                        Feature Selection
########################################################################


'''
Drop seats, keep only passengers and load. 
Cancel rate actually means the non-cancelled rate. It has a weird behaviour,
 since its mostly=1
Plotting things against distance is like getting a snapshot of the 
distribution for each route. We see that each route seems to have its
own dinamics in terms of #passengers, load, etc.
Importance of the route and passengers is correlated. Low load => low importance of the route
is_holiday does not affect at all the data

'''
#%%
#deleting columns used to calculate other columns
sel_data = full_data.copy()
del (sel_data['seats'], sel_data['departures_scheduled'], 
     sel_data['month_pas'], sel_data['year_pas'], sel_data['gdp_ori_1'],
     sel_data['gdp_dest_1'], sel_data['tot_pas_mon_ori'] 
    )
sel_data.columns

#%%
#feature engineering
sel_data['pas/flight'] = (
    sel_data['passengers']/sel_data['departures_performed']/10
    )
def fourrier_func(x):
    return 0.5 + 1/4*(1.0945*sin(2*pi/7*x) - 0.5697*cos(2*pi/7*x) + 
                      2.1256*sin(4*pi/7*x) + 0.3972*cos(4*pi/7*x) -
                      0.6531*sin(8*pi/7*x) + 1.1725*cos(8*pi/7*x)
                      )
sel_data['import_day'] = (sel_data['n_days']-15214).apply(fourrier_func)
sel_data.drop('n_days', axis=1, inplace=True)

########################################################################
#                        Preprocessing Data
########################################################################

#%%
from sklearn.preprocessing import StandardScaler
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import OneHotEncoder


# %%
#num_columns = [
#    'WeeksToDeparture', 'std_wtd', 'distance', 'departures_scheduled',
#    'departures_performed', 'seats', 'passengers', 
#    'tot_pas_mon_ori', 'import_route', 'month_pas', 'year_pas',
#    'import_month', 'load', 'cancel_rate', 'fuel_price',
#    'ratio M/F_ori', 'ratio M/F_dest', 'gdp_ori_1', 'gdp_ori_2',
#    'gdp_dest_1', 'gdp_dest_2', 'pop_metro_ori', 'pop_metro_dest',
#    'work_pop_ratio_ori', 'work_pop_ratio_dest',
#     'gdp_per_capita_ori','gdp_per_capita_dest'
#]

cat_columns = ['ori', 'dest', 'is_holiday', 'year', ]
pre_econd_col = ['month', 'day', 'weekday', 'week',
                 'import_day']
num_columns = list(sel_data.columns)
for column in pre_econd_col + cat_columns:
    num_columns.remove(column)

#%%
cat_encoder = OneHotEncoder()
num_encoder = StandardScaler()
preprocessor = make_column_transformer(
    (cat_encoder, cat_columns), 
    (num_encoder, num_columns),
    remainder='passthrough'
)

#%%
#used to perform analysis on preprocessed data only
scaled_data = preprocessor.fit_transform(sel_data)
scaled_data = pd.DataFrame(scaled_data)
########################################################################
#                        Regressor
########################################################################

#%%
import xgboost as XGB
from catboost import CatBoostRegressor
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.metrics import mean_squared_error
from sklearn.svm import SVR
from scipy.stats import uniform, randint
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import StackingRegressor


'''
List of regressors:

RandomForestRegressor
GradientBoostingRegressor
HistGradientBoostingRegressor
AdaBoostRegressor
XGBoost
CatBoost
LightGBM
SupportVectorRegressor
'''
#%%
# GB Gradient Boosting
regressor = GradientBoostingRegressor(
    learning_rate=0.14,
    n_estimators=1200,
    subsample=0.9,
    min_samples_leaf=18,
    max_depth=7
)

#%%
#XGBoost (good)
regressor = XGB.XGBRegressor( 
    gamma=0.09,
    learning_rate=0.14, 
    max_depth= 7,
    min_child_regressor=18,
    n_estimators=1200,
    reg_alpha=1,
    reg_lambda=2,
    subsample=0.9
)

#%%
#XGBoostMartha
regressor = XGB.XGBRegressor( 
    gamma=0.1,
    learning_rate=0.05, 
    max_depth= 7,
    min_child_regressor=18,
    n_estimators=1200,
    reg_alpha=1,
    reg_lambda=3,
    subsample=0.9
)

#%%
#CatBoost
regressor = CatBoostRegressor(
    iterations=5000
)

#%%
# Hist Gradient Boosting (Grid)
regressor = HistGradientBoostingRegressor(
    learning_rate=0.08,
    max_iter=1212,
    max_leaf_nodes=13,
    max_depth=7,
    min_samples_leaf=17,
    l2_regularization=5,
    max_bins=250
)


#%%
pipeline = make_pipeline(preprocessor, regressor)

########################################################################
#                        Assessing performance
########################################################################

#%%
pipeline.fit(sel_data, pas)
score = np.sqrt(mean_squared_error(
    y_true=pas, y_pred=pipeline.predict(sel_data)
    ))
print(f'The MSE on the train set is: {score:.5f}')



#%%
scores = cross_val_score(
    pipeline, sel_data, pas, cv=5, scoring='neg_mean_squared_error'
)

rmse_scores = np.sqrt(-scores)

print(
    f"RMSE: {np.mean(rmse_scores):.4f} +/- {np.std(rmse_scores):.4f}"
)


#%%
plt.figure(figsize=(10,8))
pipeline.fit(sel_data, pas)
feat_imp = regressor.feature_importances_
feat = num_columns + cat_columns + date_columns
res_df = pd.DataFrame({'Features': feat, 'Importance': feat_imp}).sort_values(by='Importance', ascending=False)
res_df.plot('Features', 'Importance', kind='bar', title='Feature Importances')
plt.ylabel('Feature Importance Score')
plt.show()

#%% 
#xgb hyperparameter range setting
space_xgb = {
    'n_estimators': randint(1100,1300),
    'learning_rate': uniform(0.03, 0.07),
    'gamma':uniform(0.09, 0.11),
    'subsample': uniform(0.8, 1),
    'reg_lambda': uniform(7, 9),
    'reg_alpha': uniform(7, 9),
    }

#%% 
#xgb Random search
search = RandomizedSearchCV(
    XGB.XGBRegressor(max_depth=7, min_child_regressor=18), param_distributions=space_xgb,
    n_iter=30, scoring='neg_mean_squared_error',
    n_jobs=-1, cv=8, verbose=3
    )

search.fit(X=scaled_data, y=pas)

def report_best_scores(results, n_top=3):
    for i in range(1, n_top + 1):
        candidates = np.flatnonzero(results['rank_test_score'] == i)
        for candidate in candidates:
            print("Model with rank: {0}".format(i))
            print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
                  results['mean_test_score'][candidate],
                  results['std_test_score'][candidate]))
            print("Parameters: {0}".format(results['params'][candidate]))
            print("")

report_best_scores(search.cv_results_, 1)

#search.fit(preprocessor.fit_transform(sel_data), pas)
#report_best_scores(search.cv_results_, 1)
#%%
#xgb GridSearch
from sklearn.model_selection import GridSearchCV
grid_xgb = {
    'learning_rate': [0.08, 0.10, 0.11, 0.12, 0.14],
    'gamma':[0.08, 0.09, 0.10, 0.11],
    'reg_lambda': [1, 2, 3, 4, 5],
    'reg_alpha': [1, 2, 3, 4, 5],
    }

grid_search = GridSearchCV(
    XGB.XGBRegressor(max_depth=7, min_child_regressor=18, n_iter=1200, subsample=0.9),
    param_grid=grid_xgb, n_jobs=-1, cv=5, verbose=5
)
grid_search.fit(preprocessor.fit_transform(sel_data), pas)

#%%
scores = pd.DataFrame(grid_search.cv_results_)
print(scores.sort_values(by='rank_test_score').head(1)['params'].values)

#%%
########################################################################
#                        Plotting performance
########################################################################

#%%
plot_df = sel_data.copy()
plot_df['y'] = pas
plot_df['y_hat'] = pipeline.predict(sel_data)
# %%
test_set, y_test = problem.get_test_data()
#encode dates
test_set['DateOfDeparture'] = pd.to_datetime(test_set['DateOfDeparture'])
test_encod = encode_dates(test_set, 'DateOfDeparture')

#solve name discrep
test_encod.rename(columns={'Departure':'ori', 'Arrival':'dest'},
                  inplace=True)


full_test = test_encod.merge(add_data, 
                             on=['year', 'month', 'day','ori', 'dest'],
                             how='left')
sel_test = full_test.copy()
del (sel_test['seats'], sel_test['departures_scheduled'], 
     sel_test['month_pas'], sel_test['year_pas'], sel_test['gdp_ori_1'],
     sel_test['gdp_dest_1'], sel_test['tot_pas_mon_ori'], 
    )
sel_test.columns

#feature engineering
sel_test['route'] = (sel_test['ori'].astype(str) + '-' + 
                     sel_test['dest'].astype(str))
sel_test['pas/flight'] = (
    sel_test['passengers']/sel_test['departures_performed']/10
    )

sel_test['import_day'] = (sel_test['n_days']-15214).apply(fourrier_func)

sel_test['y'] = y_test
sel_test['y_hat'] = pipeline.predict(sel_test)

#print test score
print(np.sqrt(mean_squared_error(y_pred=pipeline.predict(sel_test), y_true=y_test)))

# %%

########################################################################
#                        Data Visualization
########################################################################

#%%
test_sel = plot_df.loc[:, [
    'ori', 'dest', 'month', 'day', 'year', 'passengers',
    'y', 'departures_performed'
    ]]
#indexing in chronological order
test_sel['date'] = (test_sel['day'].astype(str) + '-' +
                    test_sel['month'].astype(str) + '-' 
                    + test_sel['year'].astype(str))
test_sel['date'] = pd.to_datetime(test_sel['date'])
test_sel.set_index('date', inplace=True)
test_sel.sort_index(inplace=True)

#adding each route
test_sel['route'] = test_sel['ori'] + '-' + test_sel['dest']

#pas/flight

test_sel['pas/fli'] = test_sel.passengers/test_sel.departures_performed/10

# %%
for origin in test_sel.ori.unique():
    fig = plt.figure(figsize=(12, 8))
    for route in test_sel.loc[test_sel['ori']==origin,'route'].unique():
        test1 = test_sel['ori']==origin
        test2 = test_sel['route']==route
        plt.plot(test_sel.loc[test1 & test2, 'y'], 
                    label=route)
        plt.plot(test_sel.loc[test1 & test2, 'passengers'], 
            label=route)
    plt.legend()

# %%
for route in test_sel.loc[test_sel['ori']=='DFW','route'].unique():
    fig, ax1= plt.subplots(figsize=(9, 6))
    test1 = test_sel['ori']=='DFW'
    test2 = test_sel['route']==route
    ax1.plot(test_sel.loc[test1 & test2, 'y'], 
                label='y', color='red')
    ax2 = ax1.twinx()
    ax2.plot(test_sel.loc[test1 & test2, 'pas/fli'], 
        label='pass/fli')
    plt.legend()

#%%
########################################################################
#                        Data Visualization
########################################################################

#%%
sel_data_2 = full_data.loc[:, 
    ['ori', 'dest', 'year', 'month', 'day', 'weekday', 'week', 
     'passengers', 'departures_performed', 'import_month', 'load']
]

#include routes
sel_data_2['route'] = sel_data_2['ori'] + '-' + sel_data_2['dest']

#feat. eng.
sel_data_2['pas/fli'] = sel_data_2['passengers']/sel_data_2['departures_performed']/10
sel_data_2.drop(['passengers', 'departures_performed'], axis=1,
                inplace=True)

cat_columns = ['route', 'ori', 'dest']
date_columns = ['year', 'month', 'day', 'weekday', 'week']
num_columns = list(sel_data_2.columns)
for column in cat_columns + date_columns:
    num_columns.remove(column)
#%%
pipeline.fit(sel_data_2, pas)
score = mean_squared_error(y_true=pas, y_pred=pipeline.predict(sel_data_2))
print(f'The MSE on the train set is: {score:.5f}')



#%%
scores = cross_val_score(
    pipeline, sel_data_2, pas, cv=5, scoring='neg_mean_squared_error'
)

rmse_scores = np.sqrt(-scores)

print(
    f"RMSE: {np.mean(rmse_scores):.4f} +/- {np.std(rmse_scores):.4f}"
)


#%%
plt.figure(figsize=(10,8))
pipeline.fit(sel_data_2, pas)
feat_imp = regressor.feature_importances_
feat = num_columns + cat_columns + date_columns
res_df = pd.DataFrame({'Features': feat, 'Importance': feat_imp}).sort_values(by='Importance', ascending=False)
res_df2 = res_df
res_df2.plot('Features', 'Importance', kind='bar', title='Feature Importances')
plt.ylabel('Feature Importance Score')
plt.show()

# %%
plot_df = sel_data.copy()
plot_df['y'] = pas
time_sorted_df = plot_df.sort_values(by='n_days')
day1 = time_sorted_df['n_days'] >= 15600
day2 = time_sorted_df['n_days'] <= 15614
print_df = time_sorted_df.loc[day1 & day2, :]
print_df = print_df.groupby('n_days').sum()
print_df.plot(y='passengers')
plt.set_x_ticks = range(15400, 15428, 7)

##########################################
#  Attempt feature selection
##########################################
#%%
# Selecting features for NN
from scipy.stats import spearmanr
from sklearn.feature_selection import SelectKBest
# build preprocessing pipeline
cat_columns_nn = ['ori', 'dest', 'is_holiday', "year", 
                  "month", "day","weekday", "week", 'route']
cat_data_nn = sel_data[cat_columns_nn]

encod_cat_data_nn = OrdinalEncoder(categories='auto').fit_transform(cat_data_nn)
scaled_cat_data_nn = OneHotEncoder(handle_unknown='ignore').fit_transform(encod_cat_data_nn)
scaled_num_data_nn = StandardScaler().fit_transform(num_data_nn)

#%%
from sklearn.feature_selection import f_regression
selector = SelectKBest(score_func=f_regression, k=58)
selected_columns = selector.fit_transform(scaled_cat_data_nn, pas)

##########################################
#  Attempt Ensemble Learning
##########################################
#%%
#Voting Regressor
from sklearn.ensemble import VotingRegressor
xgb = XGB.XGBRegressor(gamma=0.09, learning_rate=0.14, max_depth= 7,
                           min_child_regressor=18, n_estimators=1200, reg_alpha=1, reg_lambda=2,
                           subsample=0.9)
hist = HistGradientBoostingRegressor(learning_rate=0.08,
    max_iter=1212, max_leaf_nodes=13, max_depth=7, min_samples_leaf=17,
    l2_regularization=5, max_bins=250)

#regressor = VotingRegressor([('xbg', xgb_reg), ('hist', hist_reg)], verbose=1)

# %%
#StackingRegressor
from sklearn.ensemble import StackingRegressor

regressor = StackingRegressor([('xgb', XGB.XGBRegressor()), ('hist', HistGradientBoostingRegressor())], verbose=1)
# %%
params = {
    'xgb__gamma':uniform(0.03, 0.12),
    'xgb__learning_rate':uniform(0.01, 0.09), 
    'xgb__max_depth': randint(5, 10),
    'xgb__n_estimators':randint(800, 2500),
    'xgb__reg_alpha':uniform(2, 15),
    'xgb__reg_lambda':uniform(2, 15),
    'xgb__subsample':uniform(0.6, 0.99),
    'hist__learning_rate':uniform(0.03, 0.10),
    'hist__max_iter':randint(800, 2500), 
    'hist__max_leaf_nodes':randint(7, 20), 
    'hist__max_depth':randint(5,8), 
    'hist__min_samples_leaf':randint(8,16),
    'hist__l2_regularization':uniform(2, 15), 
    'hist__max_bins':randint(180, 255)
}

search = RandomizedSearchCV(
    estimator=regressor, param_distributions=params,
    n_iter=300, scoring='neg_mean_squared_error',
    n_jobs=-1, cv=5, verbose=3
    )

search.fit(X=scaled_data, y=pas)

def report_best_scores(results, n_top=3):
    for i in range(1, n_top + 1):
        candidates = np.flatnonzero(results['rank_test_score'] == i)
        for candidate in candidates:
            print("Model with rank: {0}".format(i))
            print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
                  results['mean_test_score'][candidate],
                  results['std_test_score'][candidate]))
            print("Parameters: {0}".format(results['params'][candidate]))
            print("")

report_best_scores(search.cv_results_, 1)



# %%
