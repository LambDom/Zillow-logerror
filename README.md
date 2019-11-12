# Zillow-logerror

# 'Allo. Je ecrive dans le readme. 

welcome to LambDom's project to predict expected logerror of zestimate values.
This project was inspired by a caggle competition posted by zillow to identify features driving error in their Zestimate.


## Project Goals:
1) create a model that can predict error in Zestimated values
2) from that model, identify key features driving *logerror*
3) clearly communicate finding to classmates

## Project Requirements:
1) utilize clustering algorithms at some point in the pipeline
2) utilize statistical testing to identify key features
3) provide helpful visulatizations explaining exploration process
4) use scaling methods on data and document why they we're used
5) impute missing values and document
6) encode cateorical data
7) feature engineering and document rational behind it

## Deliverables
1) Project Notebook showing the data pipeline process
2) Supporting .py files required to run notebook
3) Verbal presentation of finding to class

### Data Dictionary: 
- bathroom_cnt: Number of Bathrooms
- bedroom_cnt: Number of Bedrooms
- square_feet: Square Footage of Home
- lot_sqr_ft: Square Footage of Lot
- building_value: tax value of structures
- total_value: total tax value
- land_value: tax value of lot land
- tax_amount: amount of property tax paid monthly
- has_garage: boolean value of property having a garage
- multistory: boolean value of if the property has > 1 story
- bed_bath_cnt: sum of bedroom and bathrooms
- home_age: age of house at assessment in 2016
- cluster_labels: labels of clusters applied to data
- logerror: error in zestimate of home price on logarythmic scale
- latitude: Latitude of property
- longitude: Longitude of property
- abs_logerror: absolute value of error in zestimate on logarythmic scale
- los_angeles: 1 if property is in Los Angeles County
- orange: 1 if property is in Orange County
- ventura: 1 if property is in Ventura County
- abs_baseline: average of abs_logerror values
- log_baseline: average of logerror values
​