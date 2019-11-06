'''
preparation functions
'''
import numpy as np
import pandas as pd

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


def find_upper_outliers(column):
    '''
    Give it a Pandas Series/Column. This will return a a Series of values that 
    are 1.5X above the .75 quantile
    '''
    #Using the quantile function of a Series, lower and upper side of the IQR box is defined.
    q1, q3 = column.quantile([.25, .75])
    iqr = q3 - q1
    #IQR is all the values in the box made bewteen .25-.75 of the data. The middle 50.
    
    return column[column < (q3 + 1.5*iqr)]

def find_lower_outliers(column):
    '''
    Give it a Pandas Series/Column. This will return a Series of values that 
    are 1.5X above the .75 quantile
    '''
    #Using the quantile function of a Series, lower and upper side of the IQR box is defined.
    q1, q3 = column.quantile([.25, .75])
    iqr = q3 - q1
    #IQR is all the values in the box made bewteen .25-.75 of the data. The middle 50.
    
    return column[column < (q1 - 1.5*iqr)]