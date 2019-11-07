import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split

import acquire
import summarize
import prepare

import seaborn as sns
import matplotlib.pyplot as plt

# ignore warnings
import warnings
warnings.filterwarnings("ignore")


#ACQUIRE
#########

# df = pd.read_csv('data.csv')

# #The Master DataFrame of the Latitude and Longitude
# lat_long = df[['latitude','longitude']]

def get_data():
    df = pd.read_csv('data.csv')
    lat_long = df[['latitude','longitude']]
    return df, lat

#CLEAN
######

#The columns that have to do with the building themselves. No location data except for lat-long.
#12 columns
# house_vars = ['fullbathcnt','bathroomcnt','bedroomcnt','calculatedfinishedsquarefeet',
#             'heatingorsystemtypeid','lotsizesquarefeet','yearbuilt','structuretaxvaluedollarcnt', 
#             'taxvaluedollarcnt','latitude','longitude','logerror']

# #Impute the mode in heatingorsystemtypeid. Dropped null records after that
# structures_df = df[house_vars]
# structures_df.heatingorsystemtypeid.fillna(2, inplace=True)
# structures_df.dropna(inplace=True)

def make_structure_data(df):
    house_vars = ['fullbathcnt','bathroomcnt','bedroomcnt','calculatedfinishedsquarefeet',
            'heatingorsystemtypeid','lotsizesquarefeet','yearbuilt','structuretaxvaluedollarcnt', 
            'taxvaluedollarcnt','latitude','longitude','logerror']
    structures_df = df[house_vars]
    structures_df.heatingorsystemtypeid.fillna(2, inplace=True)
    structures_df.dropna(inplace=True)
    return structures_df

#PREPARE
########

def scale_and_prune_data(df):
    """
    Takes a dataframe and returns it using standard scaling and removes the high outliers in house & lot size.
    """
    train, test = train_test_split(structures_df, train_size = .8, random_state = 123)
    train.drop(['latitude','longitude'],axis=1,inplace=True)
    test.drop(['latitude','longitude'],axis=1,inplace=True)
    standard_train, standard_test, standard_object = prepare.standardize_train_test(train, test)
    standard_train = prepare.remove_upper_outliers(standard_train.calculatedfinishedsquarefeet, train)
    standard_train = prepare.remove_upper_outliers(standard_train.lotsizesquarefeet, train)
    standard_test = prepare.remove_upper_outliers(standard_test.calculatedfinishedsquarefeet, train)
    standard_test = prepare.remove_upper_outliers(standard_test.lotsizesquarefeet, train)

    return standard_train, standard_test, standard_object

# train, test = train_test_split(structures_df, train_size = .8, random_state = 123)

# #Get rid of the lat and long before the scaling happens. lat_long holds this information. We set that in ACQUIRE phase
# train.drop(['latitude','longitude'],axis=1,inplace=True)
# test.drop(['latitude','longitude'],axis=1,inplace=True)

# standard_train, standard_test, standard_object = prepare.standardize_train_test(train, test)

# #no_outliers == standard_train without upper outiers for big houses or big lots
# no_outliers = prepare.remove_upper_outliers(standard_train.calculatedfinishedsquarefeet, train)
# no_outliers = prepare.remove_upper_outliers(standard_train.lotsizesquarefeet, train)


#MODELING
#########

# kmean = KMeans(n_clusters=3)
# kmean.fit(no_outliers)

# predictions3 = kmean.labels_
# no_outliers['cluster_labels3'] = predictions3

# #THESE SHOW SOME RESULTS ABOUT THE CLUSTERS. Need to package them into functions.
# # np.unique(predictions3, return_counts=True)
# # standard_train.groupby('cluster_labels3').mean()

# no_outliers[['latitude','longitude']] = lat_long

def make_clusters(df, n_clusters=3):
    """
    This will take a dataframe and return it with a new column with cluster labels. The default it for 3
    clusters, but it can have another value given as a parameter.
    """
    kmean = KMeans(n_clusters)
    kmean.fit(df)
    predictions = kmean.labels_
    df['cluster_labels'] = predictions
    return df

# predictions3 = kmean.labels_
# no_outliers['cluster_labels3'] = predictions3

#VISUALS
##########

def show_clusters_on_map(df):
    sns.scatterplot(data=df, x='longitude', y='latitude', hue='cluster_labels3')

def show_data_on_map(df):
    sns.scatterplot(data=df, x='longitude', y='latitude')
