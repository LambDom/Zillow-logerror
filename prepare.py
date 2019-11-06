'''
preparation functions
'''
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler

def handle_missing_values(df, prop_required_column= .95, prop_required_row = .8):

    '''
    def handle_missing_values(df, prop_required_column= .95, prop_required_row = .8):


    iterates through dataframe collumns and removes all without the required percent of non nulls specified by prop_required_column, then repeats process on rown using prop_required_row

    args:
    df - dataframe to be referenced
    prop_required_column - proportion threshold of columns' non-nulls to be accepted
    prop_required_row - proportion threshold of rows' non-nulls to be accepted

    returns:
    transformed dataframe
    '''

    count_missing = df.isnull().sum()
    average_missing = count_missing/df.shape[0]
    missing_columns = pd.DataFrame({'num_rows_missing': count_missing, 'pct_rows_missing': average_missing})
    features = missing_columns[missing_columns.pct_rows_missing < (1-prop_required_column)].index.tolist()
    
    df = df[features]

    thresh = int(df.shape[1]*prop_required_row) + 1
    df = df.dropna( thresh = thresh )
    
    return df


# def standard_scaler(train, test):
#     '''
#     def standard_scaler(train, test):
    
#     reveives train and test dataframes and returns their standard scalar transformations along with their scalar object for reference later
#     '''
#     scaler_object = StandardScaler(copy=True, 
#                                    with_mean=True, 
#                                    with_std=True).fit(train) 
#     scaled_train = apply_object(train, scaler_object)
#     scaled_test  = apply_object(test,  scaler_object)
#     return  scaled_train, scaled_test, scaler_object

def remove_upper_outliers(column, df):
    '''
    Give it a Pandas Series/Column, and the DataFrame it came from. 
    This will return the dataframe without the outliers above the upperbound.
    '''
    #Using the quantile function of a Series, lower and upper side of the IQR box is defined.
    q1, q3 = column.quantile([.25, .75])
    iqr = q3 - q1
    #IQR is all the values in the box made bewteen .25-.75 of the data. The middle 50.
    upper_bound = q3 + (iqr*1.5)
    return df[column > upper_bound]

def remove_lower_outliers(column):
    '''
    Give it a Pandas Series/Column. This will return a Series of values that 
    are 1.5X above the .75 quantile
    '''
    #Using the quantile function of a Series, lower and upper side of the IQR box is defined.
    q1, q3 = column.quantile([.25, .75])
    iqr = q3 - q1
    #IQR is all the values in the box made bewteen .25-.75 of the data. The middle 50.
    lower_bound = q3 + (iqr*1.5)
    return df[column < lower_bound]

def standardize_train_test(train, test):
    """
    Uses the sklearn StandardScaler. The train and test data are given as parameters. Scaler is fit to the
    feature training dataset. Test and train are each transformed by this fitted StandardScaler. The scaled
    version of train and test are returned, along with the scaler object that made their transformation.
    """
    scaler = StandardScaler(copy=True, with_mean=True, with_std=True).fit(train)
    train_scaled = pd.DataFrame(scaler.transform(train), columns=train.columns.values).set_index([train.index.values])
    test_scaled = pd.DataFrame(scaler.transform(test), columns=test.columns.values).set_index([test.index.values])
    return train_scaled, test_scaled, scaler

def minmax_scale_train_test(train, test, minmax_range=(0,1)):
    """
    Uses the sklearn MinMaxScaler. The train and test data are given as parameters. Scaler is fit to the
    feature training dataset. Test and train are each transformed by this fitted MinMaxScaler. The scaled
    version of train and test are returned, along with the scaler object that made their transformation.
    """
    scaler = MinMaxScaler(copy=True, feature_range=minmax_range).fit(train)
    train_scaled = pd.DataFrame(scaler.transform(train), columns=train.columns.values).set_index([train.index.values])
    test_scaled = pd.DataFrame(scaler.transform(test), columns=test.columns.values).set_index([test.index.values])
    return scaler, train_scaled, test_scaled


# def remove_outliers_iqr(df, columns):
#     for col in columns:
#         q75, q25 = np.percentile(df[col], [75,25])
#         ub = 3*stats.iqr(df[col]) + q75
#         lb = q25 - 3*stats.iqr(df[col])
#         df = df[df[col] <= ub]
#         df = df[df[col] >= lb]
#     return df