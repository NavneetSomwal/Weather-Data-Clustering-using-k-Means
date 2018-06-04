# -*- coding: utf-8 -*-
"""
Created on Mon Jun  4 15:50:01 2018

@author: Navneet
"""

#Importing the Necessary Libraries
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import utils
import pandas as pd
import numpy as np
from itertools import cycle, islice
import matplotlib.pyplot as plt
from pandas.tools.plotting import parallel_coordinates

%matplotlib inline

#Creating a Pandas DataFrame from a CSV file
data = pd.read_csv('./weather/minute_weather.csv')
data.shape
data.head()

#Data Sampling(Lots of rows, so let us sample down by taking every 10th row. )
sampled_df = data[(data['rowID'] % 10) == 0]
sampled_df.shape

#Statistics 
sampled_df.describe().transpose()
sampled_df[sampled_df['rain_accumulation'] == 0].shape
sampled_df[sampled_df['rain_duration'] == 0].shape

#Drop all the Rows with Empty rain_duration and rain_accumulation 
del sampled_df['rain_accumulation']
del sampled_df['rain_duration']

rows_before = sampled_df.shape[0]
sampled_df = sampled_df.dropna()
rows_after = sampled_df.shape[0]

#How many rows did we drop ? 
rows_before - rows_after
sampled_df.columns

#Select Features of Interest for Clustering 
features = ['air_pressure', 'air_temp', 'avg_wind_direction', 'avg_wind_speed', 'max_wind_direction', 
        'max_wind_speed','relative_humidity']
select_df = sampled_df[features]
select_df.columns
select_df

#Scale the Features using StandardScaler 
X = StandardScaler().fit_transform(select_df)
X
kmeans = KMeans(n_clusters=12)
model = kmeans.fit(X)
print("model\n", model)

#What are the centers of 12 clusters we formed ? 
centers = model.cluster_centers_
centers

#Plots
#creating some utility functions first which will help us in plotting graphs:
# Function that creates a DataFrame with a column for Cluster Number

def pd_centers(featuresUsed, centers):
	colNames = list(featuresUsed)
	colNames.append('prediction')

	# Zip with a column called 'prediction' (index)
	Z = [np.append(A, index) for index, A in enumerate(centers)]

	# Convert to pandas data frame for plotting
	P = pd.DataFrame(Z, columns=colNames)
	P['prediction'] = P['prediction'].astype(int)
	return P

# Function that creates Parallel Plots

def parallel_plot(data):
	my_colors = list(islice(cycle(['b', 'r', 'g', 'y', 'k']), None, len(data)))
	plt.figure(figsize=(15,8)).gca().axes.set_ylim([-3,+3])
	parallel_coordinates(data, 'prediction', color = my_colors, marker='o')
    
P = pd_centers(features, centers)
P

#Dry Days
parallel_plot(P[P['relative_humidity'] < -0.5])


#Warm Days
parallel_plot(P[P['air_temp'] > 0.5])

#Cool Days
parallel_plot(P[(P['relative_humidity'] > 0.5) & (P['air_temp'] < 0.5)])
