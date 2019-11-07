'''
exploration functions

'''
import numpy as np
import pandas as pd 
from scipy.stats import norm
import matplotlib.pyplot as plt


def compare_standard_curves(df, target = 'logerror'):
    '''
    compare_standard_curves(df, target = 'logerror'):

    made to explore difference in mean logerror for zillow data, takes in a dataframe and a string target variable and grabs every column with less than 10 unique values and compares the mean logerror of each subset via normal curves

    args:
    df: dataframe to be used, must contain target variable
    target: string column name of target variable

    returns:
    None
    '''
    for col in df.columns.tolist():
        if df[col].nunique() < 11:
            for i in df[col].unique().tolist():
                mean = df[df[col] == i][target].mean()
                std = df[df[col] == i][target].std()
                x = np.linspace(mean-.5,mean+.5,1000)
                y = norm.pdf(x, loc=mean, scale=std)    # for example
                plt.plot(x,y, label = i)
                plt.title(col)
            plt.legend()
            
            plt.show()