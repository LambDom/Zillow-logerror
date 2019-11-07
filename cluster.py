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

def show_clusters_on_map():


    #ACQUIRE
    #########

    df = pd.read_csv('data.csv')

    #The Master DataFrame of the Latitude and Longitude
    lat_long = df[['latitude','longitude']]


    #CLEAN
    ######

    #The columns that have to do with the building themselves. No location data except for lat-long.
    #12 columns
    house_vars = ['fullbathcnt','bathroomcnt','bedroomcnt','calculatedfinishedsquarefeet',
                'heatingorsystemtypeid','lotsizesquarefeet','yearbuilt','structuretaxvaluedollarcnt', 
                'taxvaluedollarcnt','latitude','longitude','logerror']

    #Impute the mode in heatingorsystemtypeid. Dropped null records after that
    structures_df = df[house_vars]
    structures_df.heatingorsystemtypeid.fillna(2, inplace=True)
    structures_df.dropna(inplace=True)

    #PREPARE
    ########

    train, test = train_test_split(structures_df, train_size = .8, random_state = 123)

    #Get rid of the lat and long before the scaling happens. lat_long holds this information. We set that in ACQUIRE phase
    train.drop(['latitude','longitude'],axis=1,inplace=True)
    test.drop(['latitude','longitude'],axis=1,inplace=True)

    standard_train, standard_test, standard_object = prepare.standardize_train_test(train, test)

    #no_outliers == standard_train without upper outiers for big houses or big lots
    no_outliers = prepare.remove_upper_outliers(standard_train.calculatedfinishedsquarefeet, train)
    no_outliers = prepare.remove_upper_outliers(standard_train.lotsizesquarefeet, train)


    #MODELING
    #########

    kmean = KMeans(n_clusters=3)
    kmean.fit(no_outliers)

    predictions3 = kmean.labels_
    no_outliers['cluster_labels3'] = predictions3

    #THESE SHOW SOME RESULTS ABOUT THE CLUSTERS. Need to package them into functions.
    # np.unique(predictions3, return_counts=True)
    # standard_train.groupby('cluster_labels3').mean()

    no_outliers[['latitude','longitude']] = lat_long

    sns.scatterplot(data=no_outliers, x='longitude', y='latitude', hue='cluster_labels3')