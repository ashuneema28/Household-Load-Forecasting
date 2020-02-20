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
#
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM

energy = pd.read_csv('MAC000216_With_Acorn.csv')
energy = energy.reset_index()
energy.day = pd.to_datetime(energy.day,format='%Y-%m-%d').dt.date
#print(energy.head(10))

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

weather_energy =  energy.merge(weather,on='day')
#weather_energy.to_csv("weather_energy_MAC000212.csv")
#print(weather_energy.head(10))

# fig, ax1 = plt.subplots(figsize = (20,5))
# ax1.plot(weather_energy.day, weather_energy.temperatureMax, color = 'tab:orange')
# ax1.plot(weather_energy.day, weather_energy.temperatureMin, color = 'tab:pink')
# ax1.set_ylabel('Temperature')
# ax1.legend()
# ax2 = ax1.twinx()
# ax2.plot(weather_energy.day,weather_energy.energy_sum,color = 'tab:blue')
# ax2.set_ylabel('Average Energy/Household',color = 'tab:blue')
# ax2.legend(bbox_to_anchor=(0.0, 1.02, 1.0, 0.102))
# plt.title('Energy Consumption and Temperature')
# fig.tight_layout()
# plt.show()

'''clustering'''
scaler = MinMaxScaler()
weather_scaled = scaler.fit_transform(weather_energy[['temperatureMax','humidity','windSpeed']])

Nc = range(1, 20)
kmeans = [KMeans(n_clusters=i) for i in Nc]
kmeans

score = [kmeans[i].fit(weather_scaled).score(weather_scaled) for i in range(len(kmeans))]

kmeans = KMeans(n_clusters=3, max_iter=600, algorithm = 'auto')
kmeans.fit(weather_scaled)
weather_energy['weather_cluster'] = kmeans.labels_

'''adding holidays'''
holiday = pd.read_csv('C:/Users/A02290684/Desktop/clean energy/Project/data/uk_bank_holidays.csv')
holiday['Bank holidays'] = pd.to_datetime(holiday['Bank holidays'],format='%Y-%m-%d').dt.date

weather_energy = weather_energy.merge(holiday, left_on = 'day',right_on = 'Bank holidays',how = 'left')
weather_energy['holiday_ind'] = np.where(weather_energy['Bank holidays'].isna(),0,1)

'''Training'''
weather_energy['Year'] = pd.DatetimeIndex(weather_energy['day']).year
weather_energy['Month'] = pd.DatetimeIndex(weather_energy['day']).month
weather_energy.set_index(['day'],inplace=True)

'''splitting'''
model_data = weather_energy[['energy_sum','weather_cluster','holiday_ind','Acorn_value1','Acorn_value2','Acorn_value3','Acorn_value4']]
train = model_data.iloc[0:(len(model_data)-30)]
test = model_data.iloc[len(train):(len(model_data)-1)]
# train['avg_energy'].plot(figsize=(25,4))
# test['avg_energy'].plot(figsize=(25,4))
# plt.show()
'''Getting holiday week'''

# holiday_week=['2013-12-22','2013-12-23','2013-12-24','2013-12-25','2013-12-26','2013-12-27','2013-12-28','2013-12-29']
# holiday_week_train = weather_energy.loc[weather_energy['day'].isin(holiday_week)]
# holiday_week_train = holiday_week_train[['energy_sum','weather_cluster','holiday_ind','Acorn_value1','Acorn_value2','Acorn_value3','Acorn_value4']]


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
# train=train.tail(40)
# train=train.iloc[5:15]
# train = train.iloc[3:]
#train['energy_sum'].plot(figsize=(25,10),color = 'red')
#train['predicted'].plot()

# x_days = ['Sat', 'Sun', 'Mon', 'New Year', 'Wed', 'Thu', 'Fri' ]
# x = [0,1,2,3,4,5,6]
# plt.xticks(x, x_days)
# plt.plot(x,train['energy_sum'])
# plt.plot(x,train['predicted'])
# plt.yticks(np.arange(0,15,1))
# red_patch = mpatches.Patch(color='blue', label='Average Energy')
# blue_patch = mpatches.Patch(color='red', label='Predicted Energy')
# plt.legend(handles=[red_patch,blue_patch])
# plt.ylabel("Energy Consumption")
# plt.xlabel("Day")
# plt.title("Holiday week prediction of energy consumption")
# plt.show()

'''Scatter Plot Actual Vs Predicted'''

predict = model_fit.predict(start = 0,end = len(train)-1,exog = sm.add_constant(train[['weather_cluster','holiday_ind']]))
train['predicted'] = predict.values
plt.scatter(train['energy_sum'],train['predicted'],s=7,color='red',)
plt.ylim(ymax=60)
plt.yticks(np.arange(0,70,10))
plt.title("Actual Values vs Predicted Values for Individual Household")
plt.ylabel("Predicted values")
plt.xlabel("Actual values")
plt.show()

'''Test 8 Days Predicition'''
# predict = model_fit.predict(start = len(train),end = len(train)+len(test)-1,exog = sm.add_constant(test[['weather_cluster','holiday_ind','Acorn_value1','Acorn_value2','Acorn_value3','Acorn_value3']]))
# test['predicted'] = predict.values
# print(len(test))
# test = test.head(8)
#
# test['energy_sum'].plot(figsize=(25,10),color = 'green')
# test['predicted'].plot()
# plt.yticks(np.arange(0,15,1))
# red_patch = mpatches.Patch(color='blue', label='Average Energy')
# blue_patch = mpatches.Patch(color='green', label='Predicted Energy')
# plt.legend(handles=[red_patch,blue_patch])
# plt.ylabel("Energy Consumption")
# plt.xlabel("Day")
# plt.show()

'''Test holiday Predicition'''
# holiday_week=['2013-12-22','2013-12-23','2013-12-24','2013-12-25','2013-12-26','2013-12-27','2013-12-28','2013-12-29']
# predict = model_fit.predict(start = 0,end = len(train)-1,exog = sm.add_constant(train[['weather_cluster','holiday_ind','Acorn_value1','Acorn_value2','Acorn_value3','Acorn_value4']]))
#
# train['predicted'] = predict.values
# train = train.reset_index()
# train.to_csv("text.csv")
# train=train.loc[train['day'].isin(holiday_week)]
#
# train['energy_sum'].plot(figsize=(25,10),color = 'green')
# train['predicted'].plot()
# plt.yticks(np.arange(0,15,1))
# red_patch = mpatches.Patch(color='blue', label='Average Energy')
# blue_patch = mpatches.Patch(color='green', label='Predicted Energy')
# plt.legend(handles=[red_patch,blue_patch])
# plt.ylabel("Energy Consumption")
# plt.xlabel("Day")
# plt.show()

# predict = model_fit.predict(start = 0,end = len(train)-1,exog = sm.add_constant(train[['weather_cluster','holiday_ind','Acorn_value1','Acorn_value2','Acorn_value3','Acorn_value4']]))
# train['predicted'] = predict.values
# train['energy_sum'].plot(figsize=(25,10),color = 'red')
# train['predicted'].plot()
# red_patch = mpatches.Patch(color='blue', label='Average Energy')
# blue_patch = mpatches.Patch(color='red', label='Predicted Energy')
# plt.legend(handles=[red_patch,blue_patch])
# plt.ylabel("Energy Consumption")
# plt.xlabel("Day")
# plt.title("Prediction When Holiday is in the week")
# plt.show()



