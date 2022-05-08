import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Activation
print(tf.test.gpu_device_name())
print(tf.__version__)
print(tf.config.list_physical_devices())
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

x_scaler = MinMaxScaler()
y_scaler = MinMaxScaler()
pred_length = 12

col_list = ['unix', 'symbol', 'open', 'high', 'low', 'close', 'Volume BTC', 'Volume USDT', 'tradecount','values','mindate']
df = pd.read_csv('./data/Binance_BTCUSDT_minute.csv', index_col='mindate', usecols=col_list,
                 low_memory=False, parse_dates=True)
df.drop(['symbol'], inplace=True, axis=1)

df = df.astype('float')

# df['close'].plot(figsize=(14, 5))
# plt.show()

data_train = df.loc['2020':'2021']['values']
data_test = df.loc['2022':]['values']
print(data_train.head())
print(data_train.tail())
print(data_test.head())
print(data_test.tail())

data_train = np.array(data_train)
data_train = data_train[::-1]
x_train, y_train = [], []

for i in range(0, len(data_train)-pred_length-1):
    x_train.append(data_train[i:i+pred_length])

for i in range(1, len(data_train)-pred_length):
    y_train.append(data_train[i:i+pred_length])

print(pd.DataFrame(x_train))
print(pd.DataFrame(y_train))
x_train, y_train = np.array(x_train), np.array(y_train)
print(x_train.shape, y_train.shape)

print(pd.DataFrame(x_train))
print(pd.DataFrame(y_train))

x_train = x_scaler.fit_transform(x_train)
y_train = y_scaler.fit_transform(y_train)
x_train, y_train = np.array(x_train), np.array(y_train)

x_train = x_train.reshape(len(x_train[:]), len(x_train[0]), 1)
print(x_train.shape)
model = Sequential()
#model.add(LSTM(units=256, activation='relu', input_shape=(pred_length, 1),return_sequences=True))
#model.add(Dense(pred_length))
#model.add(Dense(pred_length))
#model.add(Dense(pred_length))
#model.add(Dense(1))

model.add(LSTM(units = 50, return_sequences = True, input_shape=(pred_length, 1)))
model.add(Dropout(0.2))

model.add(LSTM(units = 50, return_sequences = True))
model.add(Dropout(0.2))

model.add(LSTM(units = 50, return_sequences = True))
model.add(Dropout(0.2))

model.add(LSTM(units = 50, return_sequences = True))
model.add(Dropout(0.2))

model.add(Dense(units = 1))

model.compile(loss='mse', optimizer='adam')
model.summary()
model.fit(x_train, y_train, epochs=100, batch_size=16)
model.save('saved_model/MODEL2')

data_test = np.array(data_test)
data_test = data_test[::-1]
x_test, y_test = [], []

for i in range(0, len(data_train)-pred_length-1):
    x_test.append(data_train[i:i+pred_length])

for i in range(1, len(data_train)-pred_length):
    y_test.append(data_train[i:i+pred_length])

print(pd.DataFrame(x_test))
print(pd.DataFrame(y_test))
x_test, y_test = np.array(x_test), np.array(x_test)
x_test = x_scaler.transform(x_test)
x_test = x_test.reshape(len(x_test[:]), len(x_test[0]), 1)

y_test = y_scaler.transform(y_test)
y_test = y_test.reshape(len(y_test[:]), len(y_test[0]), 1)
print(x_test.shape)
print(y_test.shape)
y_pred = model.predict(y_test)
print(y_pred.shape)
print(len(y_test[:]))
print(y_pred)
y_pred = y_pred.reshape(len(y_test[:]), len(y_test[0]))
print(y_pred.shape)
y_pred = y_scaler.inverse_transform(y_pred)
print(y_pred)
pred = y_pred[:, pred_length-1]

values = df[0:len(pred)][['close']].astype(float)
values['Predicted'] = pred
print(values)
values.rename(columns={'close': 'Real Values'}, inplace=True)
values.plot(figsize=(14, 5))
plt.show()