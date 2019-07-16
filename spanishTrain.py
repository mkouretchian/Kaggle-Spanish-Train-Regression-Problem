# -*- coding: utf-8 -*-
"""
Created on Fri Jun 28 15:30:42 2019

@author: Mahdi Kouretchian
"""

import numpy as np
import datetime
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
#########In this section the data is read from the file#####
data_frame=pd.read_csv('C:/Users/Mahdi Kouretchian/Desktop/spanish train/renfe.csv')
###### drop the first two non important columns ####
data_frame.drop(['Unnamed','insert_date'],axis=1,inplace=True)
####  Define a function to encode the location  ######
def origin_to_numerical(var):
    if var=='BARCELONA':
        var=1
    if var=='MADRID':
        var=2
    if var=='PONFERRADA':
        var=3
    if var=='SEVILLA':
        var=4
    if var=='VALENCIA':
        var=5
    return var
data_frame['origin']=data_frame['origin'].apply(origin_to_numerical)
data_frame['destination']=data_frame['destination'].apply(origin_to_numerical)
###converting to categorical data #######
data_frame['origin']=data_frame['origin'].astype('category')
data_frame['destination']=data_frame['destination'].astype('category')
### separate date from time #####
data_frame['date']=pd.to_datetime(data_frame['start_date']).dt.date
data_frame['start_time']=pd.to_datetime(data_frame['start_date']).dt.time
data_frame['end_time']=pd.to_datetime(data_frame['end_date']).dt.time
### drop start_date and end_date ###
data_frame.drop(['start_date','end_date'],axis=1,inplace=True)


### define a function that separates hour
def separate_hour(var):
    time = datetime.datetime.strptime(str(var), '%H:%M:%S')
    var=time.hour
    return var
### define a function that separates minutes ###
def separate_minute(var):
    time = datetime.datetime.strptime(str(var), '%H:%M:%S')
    var=time.minute
    return var
### separate hour and minute of start time ###
data_frame['start_time_hour']=data_frame['start_time'].apply(separate_hour)
data_frame['start_time_min']=data_frame['start_time'].apply(separate_minute)
if 'start_time_hour' in data_frame:
    print('Start hour is correct')
### separate hour and minute of end time ###
data_frame['end_time_hour']=data_frame['end_time'].apply(separate_hour)
data_frame['end_time_min']=data_frame['end_time'].apply(separate_minute)
### Give numeric value to train type ###
def train_type(var):
    if var=='ALVIA':
        var=1
    if var=='AV City':
        var=2
    if var=='AVE':
        var=3
    if var=='AVE-LD':
        var=4
    if var=='AVE-MD':
        var=5
    if var=='AVE-TGV':
        var=6
    if var=='INTERCITY':
        var=7
    if var=='LD':
        var=8
    if var=='LD-AVE':
        var=9
    if var=='LD-MD':
        var=10
    if var=='MD':
        var=11
    if var=='MD-AVE':
        var=12
    if var=='MD-LD':
        var=13
    if var=='R. EXPRES':
        var=14
    if var=='REGIONAL':
        var=15
    if var=='TRENHOTEL':
        var=16
    return var
data_frame['train_type']=data_frame['train_type'].apply(train_type)
### Converting train_type to categorical data ###
data_frame['train_type']=data_frame['train_type'].astype('category')
### Give numeric value to train class ###
def train_class(var):
    if var=='Cama G. Clase':
        var=1
    if var=='Cama Turista':
        var=2
    if var=='Preferente':
        var=3
    if var=='Turista':
        var=4
    if var=='Turista con enlace':
        var=5
    if var=='Turista Plus':
        var=6
    return var
data_frame['train_class']=data_frame['train_class'].apply(train_class)
### Converting train_class to categorical data ###
data_frame['train_class']=data_frame['train_class'].astype('category')
### Give numeric values to fare ###
def fare(var):
    if var=='Adulto ida':
        var=1
    if var=='Flexible':
        var=2
    if var=='Grupos Ida':
        var=3
    if var=='Individual-Flexible':
        var=4
    if var=='Mesa':
        var=5
    if var=='Promo':
        var=6
    if var=='Promo +':
        var=7
    return var
data_frame['fare']=data_frame['fare'].apply(fare)
### converting fare to categorical data ###
data_frame['fare']=data_frame['fare'].astype('category')
### replacing empty prices with the average ###
data_frame['price'].fillna((data_frame['price'].mean()),inplace=True) 
###drop unnecessary columns ###
data_frame.drop(['date','start_time','end_time'],axis=1,inplace=True)
data_frame['train_class'].replace('', np.nan, inplace=True)
data_frame['fare'].replace('', np.nan, inplace=True)
data_frame.dropna(inplace=True)
data_frame.to_csv("data_frame.csv",index=False) 
### initializing the linear regression model ###
X = data_frame[['origin', 'destination', 'train_type', 'train_class', 'fare',
       'start_time_hour', 'start_time_min', 'end_time_hour', 'end_time_min']]
y = data_frame['price']
### splitting the data to train test sets ###
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test=train_test_split(X,y,test_size=0.2) 
### training the model ###
from sklearn.linear_model import LinearRegression
lm = LinearRegression()
lm.fit(X_train,y_train)
### saving the linear coefficient as a dataframe ###
coeff_data_frame=pd.DataFrame( lm.coef_ , X.columns , columns=['Coeff'])
coeff_data_frame
###Evaluation#####
predictions=lm.predict(X_test)
plt.scatter(y_test,predictions)
sns.distplot((y_test-predictions))
### Error evaluation ####
from sklearn import metrics
error=np.sqrt(metrics.mean_squared_error(y_test,predictions))
