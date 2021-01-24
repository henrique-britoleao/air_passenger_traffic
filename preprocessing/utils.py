from math import pi, sin, cos
from pandas import to_datetime

def fourrier_func(x):
    return 0.5 + 1/4*(1.0945*sin(2*pi/7*x) - 0.5697*cos(2*pi/7*x) + 
                      2.1256*sin(4*pi/7*x) + 0.3972*cos(4*pi/7*x) -
                      0.6531*sin(8*pi/7*x) + 1.1725*cos(8*pi/7*x))

def encode_dates(X):
    # Encode the date information from the DateOfDeparture columns
    X.loc[:, 'year'] = X['DateOfDeparture'].dt.year
    X.loc[:, 'month'] = X['DateOfDeparture'].dt.month
    X.loc[:, 'day'] = X['DateOfDeparture'].dt.day
    X.loc[:, 'weekday'] = X['DateOfDeparture'].dt.weekday
    X.loc[:, 'week'] = X['DateOfDeparture'].dt.week
    X.loc[:, 'n_days'] = X['DateOfDeparture'].apply(
        lambda date: (date - to_datetime("1970-01-01")).days
    )
    # Finally we can drop the original columns from the dataframe
    return X.drop(columns=["DateOfDeparture"])