import pandas as pd
import numpy as np 

from sklearn.impute import SimpleImputer

import prepare
import env

# 1
# Acquire data from mySQL using the python module to connect and query. You will want to end with a single 
# dataframe. Make sure to include: the logerror, all fields related to the properties that are available. 
# You will end up using all the tables in the database.

# Be sure to do the correct join. 

#   only include properties with a transaction in 2017, and include only the last transaction for each 
#   properity (so no duplicate property id's), along with zestimate error and date of transaction.

#   only include properties that include a latitude and longitude value

def get_connection(db, user=env.user, host=env.host, password=env.password):
    return f'mysql+pymysql://{user}:{password}@{host}/{db}'

def get_real_zillow_singles():
    #SQL cndition have been increased to widdle away homes that are not unitcnt of 1 OR null
    #a minimum of 500 sqft cuts off the inhosptiable homes
    #kept the geographical requirments
    query = '''
    SELECT prop.*, pred.logerror, pred.transactiondate
    FROM predictions_2017 AS pred
    LEFT JOIN properties_2017 AS prop  USING(parcelid)
    WHERE (bedroomcnt > 0 AND bathroomcnt > 0 AND calculatedfinishedsquarefeet > 500 AND latitude IS NOT NULL AND longitude IS NOT NULL) 
    AND (unitcnt = 1 OR unitcnt IS NULL)
    ;
    '''
    df = pd.read_sql(query, get_connection('zillow'))
    df.sort_values(by='transactiondate', ascending=False)
    df.drop_duplicates(subset ="parcelid", keep = 'first', inplace = True) 
    return df

def get_zillow_singles():
    #SQL cndition have been increased to widdle away homes that are not unitcnt of 1 OR null
    #a minimum of 500 sqft cuts off the inhosptiable homes
    #kept the geographical requirments
    query = '''
    SELECT prop.*, pred.logerror, pred.transactiondate
    FROM predictions_2017 AS pred
    LEFT JOIN properties_2017 AS prop  USING(parcelid)
    WHERE (bedroomcnt > 0 AND bathroomcnt > 0 AND calculatedfinishedsquarefeet > 500 AND latitude IS NOT NULL AND longitude IS NOT NULL) 
    AND (unitcnt = 1 OR unitcnt IS NULL)
    ;
    '''
    df = pd.read_sql(query, get_connection('zillow'))
    # df.sort_values(by='transactiondate', ascending=False)
    # df.drop_duplicates(subset ="parcelid", keep = 'first', inplace = True) 
    return df

#acquires a chunk that is unique records, using most recent transactionb
def get_zillow_chunk():
    query = '''
    SELECT prop.*, pred.logerror, pred.transactiondate
    FROM predictions_2017 AS pred
    LEFT JOIN properties_2017 AS prop  USING(parcelid)
    WHERE (latitude IS NOT NULL AND longitude IS NOT NULL) 
    ;
    '''
    df = pd.read_sql(query, get_connection('zillow'))
    df.sort_values(by='transactiondate', ascending=False)
    df.drop_duplicates(subset ="parcelid", keep = 'first', inplace = True) 
    return df

def get_mall_customers():
    query = '''
    SELECT *
    FROM customers
    ;
    '''
    df = pd.read_sql(query, get_connection('mall_customers'))
    return df 

def wrangle_zillow():
    #get zillow data
    zillow = pd.read_csv('data.csv')
    
    #replace nones with nans
    zillow.fillna(value=pd.np.nan, inplace=True)
    
    #quick adds
    zillow['has_basement'] = zillow.basementsqft > 0
    zillow['has_fireplace'] = zillow.fireplacecnt > 0
    zillow['has_deck'] = ~zillow.decktypeid.isna()
    zillow['has_garage'] = zillow.garagetotalsqft > 0
    zillow['has pool_or_spa'] = (zillow.hashottuborspa == 1) | (zillow.poolcnt> 0)
    zillow['has_yardbuilding'] = (zillow.yardbuildingsqft17) > 0 | (zillow.yardbuildingsqft26 > 0)
    zillow['multistory'] = zillow.numberofstories > 1

    # drop collumns with more than 50% data missing and rows with 80% missing
    zillow = prepare.handle_missing_values(zillow, prop_required_column= .25, prop_required_row=.60)

    #rename columns
    zillow.rename(columns = {
    'parcelid': 'parcel_id',
    'airconditioningtypeid': 'ac_type_id',
    'bathroomcnt': 'bathroom_cnt',
    'bedroomcnt': 'bedroom_cnt',
    'buildingqualitytypeid': 'building_quality_type',
    'calculatedbathnbr': 'sum_bath_and_bed',
    'calculatedfinishedsquarefeet': 'square_feet',
    'fips': 'fips_code',
    'fullbathcnt': 'full_bath_cnt',
    'garagecarcnt': 'garage_car_cnt',
    'garagetotalsqft': 'garage_sqr_ft',
    'heatingorsystemtypeid': 'heating_type_id',
    'lotsizesquarefeet': 'lot_sqr_ft',
    'propertycountylandusecode': 'property_land_use_code',
    'propertylandusetypeid': 'property_land_use_id',
    'propertyzoningdesc': 'property_zoning',
    'rawcensustractandblock': 'raw_census_block',
    'regionidcity': 'city_id',
    'regionidcounty': 'county_id',
    'regionidneighborhood': 'neighborhood_id',
    'regionidzip': 'zipcode_id',
    'roomcnt': 'room_cnt',
    'unitcnt': 'unit_cnt',
    'yearbuilt': 'year_built',
    'structuretaxvaluedollarcnt': 'building_value',
    'taxvaluedollarcnt': 'total_value',
    'assessmentyear': 'year_assessed',
    'landtaxvaluedollarcnt': 'land_value',
    'taxamount': 'tax_amount',
    'censustractandblock': 'census_block',
    'transactiondate': 'transaction_date'
    }, inplace= True)

    #clean up dataframe
    #~~~~~~~~~~~~~~~~~~~
    # drop redundant columns
    zillow.drop(columns = 'raw_census_block', inplace= True)
    zillow.drop(columns = 'county_id', inplace= True)
    zillow.drop(columns = ['garage_sqr_ft', 'garage_car_cnt'], inplace= True)
    zillow.drop(columns = 'finishedsquarefeet12', inplace= True)
    #drop columns with no variance
    zillow.drop(columns = ['year_assessed', 'unit_cnt'], inplace= True)
    #change build year measure to house age and drop year_built
    zillow['home_age'] = 2016 - zillow.year_built
    zillow.drop(columns = 'year_built', inplace= True)
    #latitude and longitude need to be divided by a factor of 6
    zillow['latitude'] = zillow.latitude/(10^6)
    zillow['longitude'] = zillow.longitude/(10^6)
    #we have no reference of what building quality type means
    zillow.drop(columns = 'building_quality_type', inplace=True)

    #impute on columns with > 1,000 nulls
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #impute mode on columns with high likelyhood of occurance
    modeimputer = SimpleImputer(strategy = 'most_frequent')
    zillow[['ac_type_id', 'heating_type_id']] = modeimputer.fit_transform(zillow[['ac_type_id', 'heating_type_id']])
    #impute median on lot_sqr_ft
    medimputer = SimpleImputer(strategy= 'median')
    zillow[['lot_sqr_ft']] = medimputer.fit_transform(zillow[['lot_sqr_ft']])
    #drop columns with too many nulls and no obvious method of imputing them
    zillow.drop(columns = ['neighborhood_id', 'property_zoning', 'city_id'], inplace = True)

    #drop remaining nulls
    zillow.dropna(inplace= True)
    
    return zillow