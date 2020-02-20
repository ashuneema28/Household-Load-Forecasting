import pandas as pd
import numpy as np
from pandas import datetime
from matplotlib import pyplot as plt
import os
from matplotlib import pyplot

from statsmodels.tsa.arima_model import ARIMA
from matplotlib import pyplot
from pandas.plotting import autocorrelation_plot
import matplotlib.patches as mpatches
#from pyramid.arima import auto_arima
#from pmdarima.arima import auto_arima
import pyflux as pf
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import statsmodels.api as sm
from statsmodels.tsa.statespace.sarimax import SARIMAX
import math
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM

energy = pd.read_csv('MAC000046_With_Acorn.csv')
ar = energy['day'].tolist()
energy = energy.reset_index()
energy.day = pd.to_datetime(energy.day,format='%Y-%m-%d').dt.date

energy2 = pd.read_csv('MAC000216_With_Acorn.csv')
energy2 = energy2.loc[energy2['day'].isin(ar)]
energy2 = energy2.reset_index()
energy2.day = pd.to_datetime(energy2.day,format='%Y-%m-%d').dt.date

energy3 = pd.read_csv('MAC000213_With_Acorn.csv')
energy3 = energy3.loc[energy3['day'].isin(ar)]
energy3 = energy3.reset_index()
energy3.day = pd.to_datetime(energy3.day,format='%Y-%m-%d').dt.date

weather = pd.read_csv('C:/Users/A02290684/Desktop/clean energy/Project/data/weather_daily_darksky.csv')

weather['day']=  pd.to_datetime(weather['time']) # day is given as timestamp
weather['day']=  pd.to_datetime(weather['day'],format='%Y%m%d').dt.date
# selecting numeric variables
weather = weather[['temperatureMax', 'windBearing', 'dewPoint', 'cloudCover', 'windSpeed',
                   'pressure', 'apparentTemperatureHigh', 'visibility', 'humidity',
                   'apparentTemperatureLow', 'apparentTemperatureMax', 'uvIndex',
                   'temperatureLow', 'temperatureMin', 'temperatureHigh',
                   'apparentTemperatureMin', 'moonPhase','day']]
weather = weather.dropna()

weather_energy  = energy.merge(weather,on='day')
weather_energy2 = energy2.merge(weather,on='day')
weather_energy3 = energy3.merge(weather,on='day')

'''clustering 1'''
scaler = MinMaxScaler()
weather_scaled = scaler.fit_transform(weather_energy[['temperatureMax','humidity','windSpeed']])

Nc = range(1, 20)
kmeans = [KMeans(n_clusters=i) for i in Nc]
kmeans

score = [kmeans[i].fit(weather_scaled).score(weather_scaled) for i in range(len(kmeans))]

kmeans = KMeans(n_clusters=3, max_iter=600, algorithm = 'auto')
kmeans.fit(weather_scaled)
weather_energy['weather_cluster'] = kmeans.labels_

'''clustering 2'''
scaler = MinMaxScaler()
weather_scaled2 = scaler.fit_transform(weather_energy2[['temperatureMax','humidity','windSpeed']])

Nc = range(1, 20)
kmeans = [KMeans(n_clusters=i) for i in Nc]
kmeans

score = [kmeans[i].fit(weather_scaled2).score(weather_scaled2) for i in range(len(kmeans))]

kmeans = KMeans(n_clusters=3, max_iter=600, algorithm = 'auto')
kmeans.fit(weather_scaled2)
weather_energy2['weather_cluster'] = kmeans.labels_

'''adding holidays'''
holiday = pd.read_csv('C:/Users/A02290684/Desktop/clean energy/Project/data/uk_bank_holidays.csv')
holiday['Bank holidays'] = pd.to_datetime(holiday['Bank holidays'],format='%Y-%m-%d').dt.date

weather_energy = weather_energy.merge(holiday, left_on = 'day',right_on = 'Bank holidays',how = 'left')
weather_energy['holiday_ind'] = np.where(weather_energy['Bank holidays'].isna(),0,1)

'''adding holidays 2'''
holiday = pd.read_csv('C:/Users/A02290684/Desktop/clean energy/Project/data/uk_bank_holidays.csv')
holiday['Bank holidays'] = pd.to_datetime(holiday['Bank holidays'],format='%Y-%m-%d').dt.date

weather_energy2 = weather_energy2.merge(holiday, left_on = 'day',right_on = 'Bank holidays',how = 'left')
weather_energy2['holiday_ind'] = np.where(weather_energy2['Bank holidays'].isna(),0,1)

'''Training'''
weather_energy['Year'] = pd.DatetimeIndex(weather_energy['day']).year
weather_energy['Month'] = pd.DatetimeIndex(weather_energy['day']).month
weather_energy.set_index(['day'],inplace=True)

'''Training2'''
weather_energy2['Year'] = pd.DatetimeIndex(weather_energy2['day']).year
weather_energy2['Month'] = pd.DatetimeIndex(weather_energy2['day']).month
weather_energy2.set_index(['day'],inplace=True)

'''splitting'''
model_data = weather_energy[['energy_sum','weather_cluster','holiday_ind','Acorn_value1','Acorn_value2','Acorn_value3','Acorn_value4']]
train = model_data.iloc[0:(len(model_data)-30)]
test = model_data.iloc[len(train):(len(model_data)-1)]


'''splitting2'''
model_data2 = weather_energy2[['energy_sum','weather_cluster','holiday_ind','Acorn_value1','Acorn_value2','Acorn_value3','Acorn_value4']]
train2 = model_data2.iloc[0:(len(model_data2)-30)]
test2 = model_data2.iloc[len(train2):(len(model_data2)-1)]

result = [train,train2]
train = pd.concat(result)
result2 = [test,test2]
test = pd.concat(result2)
print(train.head(10))

'''test'''
t = sm.tsa.adfuller(train.energy_sum, autolag='AIC')
#pd.Series(t[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
def difference(dataset, interval):
    diff = list()
    for i in range(interval, len(dataset)):
        value = dataset.iloc[i] - dataset.iloc[i - interval]
        diff.append(value)
    return diff

t  = sm.tsa.adfuller(difference(train.energy_sum,1), autolag='AIC')
#pd.Series(t[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])

endog = train['energy_sum']
exog = sm.add_constant(train[['weather_cluster','holiday_ind','Acorn_value1','Acorn_value2','Acorn_value3','Acorn_value4']])

mod = sm.tsa.statespace.SARIMAX(endog=endog, exog=exog, order=(7,1,1),seasonal_order=(1,1, 0, 12),trend='c')
model_fit = mod.fit()
model_fit.summary()

#train['avg_energy'].plot(figsize=(25,10))
#model_fit.fittedvalues.plot()
#plt.show()

'''Test Prediction'''
# predict = model_fit.predict(start = len(train),end = len(train)+len(test)-1,exog = sm.add_constant(test[['weather_cluster','holiday_ind','Acorn_value1','Acorn_value2','Acorn_value3','Acorn_value3']]))
# test['predicted'] = predict.values
#
# test = test.head(15)
# test['energy_sum'].plot(figsize=(25,10),color = 'red')
# test['predicted'].plot()
# red_patch = mpatches.Patch(color='blue', label='Average Energy')
# blue_patch = mpatches.Patch(color='red', label='Predicted Energy')
# plt.legend(handles=[red_patch,blue_patch])
# plt.ylabel("Energy Consumption")
# plt.xlabel("Day")
# plt.show()

'''Train Prediction'''
# predict = model_fit.predict(start = 0,end = len(train)-1,exog = sm.add_constant(train[['weather_cluster','holiday_ind','Acorn_value1','Acorn_value2','Acorn_value3','Acorn_value3']]))
# train['predicted'] = predict.values
# print(train.tail(8))
# #train.to_csv("Acorn_merged_Prediction_Train.csv")
# train['energy_sum'].plot(figsize=(25,10),color = 'red')
# train['predicted'].plot()
# red_patch = mpatches.Patch(color='blue', label='Average Energy')
# blue_patch = mpatches.Patch(color='red', label='Predicted Energy')
# plt.legend(handles=[red_patch,blue_patch])
# plt.ylabel("Energy Consumption")
# plt.xlabel("Day")
# plt.show()

#
'''Scatter Plot Actual Vs Predicted'''

predict = model_fit.predict(start = 0,end = len(train)-1,exog = sm.add_constant(train[['weather_cluster','holiday_ind','Acorn_value1','Acorn_value2','Acorn_value3','Acorn_value4']]))
train['predicted'] = predict.values

ans1 = train.loc[train['Acorn_value4']==76]
ans2 = train.loc[train['Acorn_value4']==50]

plt.scatter(ans1['energy_sum'],ans1['predicted'],s=5,color='blue',)
plt.scatter(ans2['energy_sum'],ans2['predicted'],s=5,color='green',)
plt.ylabel("Predicted values")
plt.xlabel("Actual values")
plt.title("Actual Values Vs Predicted Values For houses of different Acorns")
plt.show()


