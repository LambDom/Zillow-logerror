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
df = pd.read_csv('data.csv')

#The Master DataFrame of the Latitude and Longitude
lat_long = df[['latitude','longitude']]

#The columns that have to do with the building themselves. No location data except for lat-long.
#12 columns
house_vars = ['fullbathcnt','bathroomcnt','bedroomcnt','calculatedfinishedsquarefeet',
              'heatingorsystemtypeid','lotsizesquarefeet','yearbuilt','structuretaxvaluedollarcnt', 
              'taxvaluedollarcnt','latitude','longitude','logerror']

structures_df = df[house_vars]