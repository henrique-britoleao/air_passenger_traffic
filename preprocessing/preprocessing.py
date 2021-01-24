###############################################################################
import os
import pandas as pd
from preprocessing.utils import fourrier_func, encode_dates
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, \
                                  OrdinalEncoder, StandardScaler
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline

class DataMerger():
    '''
    A class to package the data merging process

    Attributes 
    ----------
    external_data: pd.DataFrame
        Data frame containing the external information

    Methods
    -------
    merge(data):
        Merges the parsed data with external_data
    '''
    def __init__(self):
        filepath = os.path.join(
            os.path.dirname(__file__), 'external_data.csv'
        )   
        self.external_data = pd.read_csv(filepath)

    def merge(self, data):
        '''
        Merges the parsed data with the additional data

        Input
        ----- 
        data: pd.DataFrame
            data to be merged

        Returns 
        -------
        X_merged: pd.DataFrame
            merged data
        '''
        X = data.copy()  # to avoid raising SettingOnCopyWarning
        # Make sure that DateOfDeparture is of dtype datetime
        X.loc[:, "DateOfDeparture"] = pd.to_datetime(X['DateOfDeparture'])
        X = encode_dates(X)

        X.rename(columns={'Departure':'ori', 'Arrival':'dest'},
                  inplace=True)
        self.external_data.drop_duplicates(
            ['year', 'month', 'day','ori', 'dest'], inplace=True
        )

        X_merged = X.merge(
            self.external_data, on=['year', 'month', 'day','ori', 'dest'],
            how='left'
        )
        #delete unused columns 
        X_merged.drop(
            ['seats','departures_scheduled','month_pas','year_pas','gdp_ori_1',
            'gdp_dest_1','tot_pas_mon_ori'],
            axis=1, inplace=True
        )
        X_merged['pas/flight'] = (
            X_merged['passengers']/X_merged['departures_performed']/10
        )    
        X_merged['import_day'] = (
            X_merged['n_days']-15214).apply(fourrier_func
        )

        return X_merged
    
def preprocessor(model):
    '''
    Builds a preprocessor of the data based on given the model

    Input
    -----
    model: str
        Either 'NN' or 'Boost' for the neural network and the CatBoost 
        regressor respectively
    Returns
    -------
    pipeline: sklearn.pipeline.Pipeline
        Pipeline containing merging and preprocessing steps
    '''
    data_merger = FunctionTransformer(DataMerger.merge) # to use in sklearn 
    
    #specify categorical and numerical columns
    cat_columns = ['ori', 'dest','is_holiday', 'year', 'month', 'day', 'weekday',
                   'week']
    num_columns = ['WeeksToDeparture', 'std_wtd', 'distance', 'passengers', 
                   'departures_performed','import_route', 'import_month', 'load', 
                   'cancel_rate', 'fuel_price', 'ratio M/F_ori', 'ratio M/F_dest', 
                   'gdp_ori_2', 'gdp_dest_2', 'pop_metro_ori', 'pop_metro_dest', 
                   'work_pop_ratio_ori', 'work_pop_ratio_dest', 
                   'gdp_per_capita_ori', 'gdp_per_capita_dest', 'pas/flight', 
                   'n_days', 'import_day']

    if model == 'NN':
        cat_encoder = OneHotEncoder(handle_unknown="ignore")
    elif model == 'Boost':
        cat_encoder = OrdinalEncoder(categories='auto')
    else:
        raise TypeError(
            f"""model must be a str containing either 'NN' or 'Boost', but it  
            is a {type(model)} containing {model}"""
        )

    num_encoder = make_pipeline(StandardScaler())
    transformer = make_column_transformer(
        (cat_encoder, cat_columns), 
        (num_encoder, num_columns),
        remainder='passthrough'
    )

    return make_pipeline(data_merger, transformer)

