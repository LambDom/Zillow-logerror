import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn import metrics 
from scipy.spatial.distance import cdist 

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
    return df, lat_long

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
    house_vars = ['bathroomcnt','bedroomcnt','calculatedfinishedsquarefeet',
            'heatingorsystemtypeid','lotsizesquarefeet','yearbuilt','structuretaxvaluedollarcnt', 
            'taxvaluedollarcnt','latitude','longitude','logerror']
    structures_df = df[house_vars]
    structures_df.heatingorsystemtypeid.fillna(2, inplace=True)
    structures_df.dropna(inplace=True)
    return structures_df

#PREPARE
########

def split_data(df):
    """
    Takes a dataframe, breaks it into 80-20 training/testing sets
    """
    train, test = train_test_split(df, train_size = .8, random_state = 123)
    train.drop(['latitude','longitude'],axis=1,inplace=True)
    test.drop(['latitude','longitude'],axis=1,inplace=True)
    return train, test

def standardize_train_test(train,test):
    standard_train, standard_test, blob = prepare.standardize_train_test(train, test)
    return standard_train, standard_test, blob

def remove_upper_outliers(train, test):
    """
    Removes the upper outliers of the calculatedfinishedsquarefeet and lotsizesquarefeet
    """
    train = prepare.remove_upper_outliers(train.calculatedfinishedsquarefeet, train)
    train = prepare.remove_upper_outliers(train.lotsizesquarefeet, train)
    test = prepare.remove_upper_outliers(test.calculatedfinishedsquarefeet, test)
    test = prepare.remove_upper_outliers(test.lotsizesquarefeet, test)
    return train, test

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

def show_clusters_on_map(df, cluster_label='cluster_labels'):
    temp = pd.read_csv('data.csv')
    lat_long = temp[['latitude','longitude']]
    df[['latitude','longitude']] = lat_long
    sns.scatterplot(data=df, x='longitude', y='latitude', hue=cluster_label)

def show_data_on_map(df):
    sns.scatterplot(data=df, x='longitude', y='latitude')

def list_inertia_scores(X):
    distortions = [] 
    inertias = [] 
    mapping1 = {} 
    mapping2 = {} 
    K = range(1,10) 

    for k in K: 
        #Building and fitting the model 
        kmeanModel = KMeans(n_clusters=k).fit(X) 
        kmeanModel.fit(X)     
            
        distortions.append(sum(np.min(cdist(X, kmeanModel.cluster_centers_, 
                            'euclidean'),axis=1)) / X.shape[0]) 
        inertias.append(kmeanModel.inertia_) 

        mapping1[k] = sum(np.min(cdist(X, kmeanModel.cluster_centers_, 
                        'euclidean'),axis=1)) / X.shape[0] 
        mapping2[k] = kmeanModel.inertia_ 


    for key,val in mapping1.items(): 
        print(str(key)+' : '+str(val)) 

def my_inv_transform(scaler, train_scaled):
    df = pd.DataFrame(scaler.inverse_transform(train_scaled), columns=train_scaled.columns.values).set_index([train_scaled.index.values])
    return scaler, df

