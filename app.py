import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from keras.models import load_model
import streamlit as st
yf.pdr_override()
from pandas_datareader import data as pdr
import datetime

start = '1990-01-01'
# end = '2020-12-31'
end = datetime.datetime.today()

st.title('Stock Trend Prediction')

user_input = st.text_input('Enter Stock Ticker', 'AAPL')
df = pdr.get_data_yahoo(user_input, start, end)

# Describing data

st.subheader('Data from 1990 - Present')
st.write(df.describe())

# Visualization of data

st.subheader('Closing Price vs Time Chart')
fig = plt.figure(figsize = (12,6))
plt.plot(df.Close)
plt.xlabel('Time', color='#ffffff')
plt.ylabel('Price', color='#ffffff')
st.plotly_chart(fig, use_container_width=True)

st.subheader('Closing Price vs Time Chart with 100MA')
ma100 = df.Close.rolling(100).mean()
fig = plt.figure(figsize = (12,6))
plt.plot(df.Close, label='Close Price')
plt.plot(ma100, label='100MA')
st.plotly_chart(fig, use_container_width=True)

st.subheader('Closing Price vs Time Chart with 100MA and 200MA')
ma100 = df.Close.rolling(100).mean()
ma200 = df.Close.rolling(200).mean()
fig = plt.figure(figsize = (12,6))
plt.plot(df.Close, label='Close Price')
plt.plot(ma100, label='100MA')
plt.plot(ma200, label='200MA')
st.plotly_chart(fig, use_container_width=True)


# Splitting data into testing and training

data_training = pd.DataFrame(df['Close'][0:int(len(df)*0.75)])
data_testing = pd.DataFrame(df['Close'][int(len(df)*0.75):int(len(df))])

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0,1))

data_training_array = scaler.fit_transform(data_training)

# Loading model

model = load_model('keras_model.h5')

# Testing 

past_100_days = data_training.tail(100)
final_df = pd.concat([past_100_days, data_testing], ignore_index=True)
input_data = scaler.fit_transform(final_df)

x_test = []
y_test = []

for i in range(100, input_data.shape[0]):
    x_test.append(input_data[i-100:i])
    y_test.append(input_data[i, 0])

x_test, y_test = np.array(x_test), np.array(y_test)

y_predicted = model.predict(x_test)
scaler = scaler.scale_

scale_factor = 1/scaler[0]
y_predicted = y_predicted * scale_factor
y_test = y_test * scale_factor

# Final graph

st.subheader('Predictions vs Original Trend')
fig2 = plt.figure(figsize=(12,6))
plt.plot(y_test, 'b', label = 'Original Price Trend')
plt.plot(y_predicted, 'r', label = 'Predicted Price Trend')
# plt.xlabel('Time', color='#ffffff')
# plt.ylabel('Price', color='#ffffff')
plt.legend()
st.plotly_chart(fig2, use_container_width=True)