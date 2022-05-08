import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from pandas.tseries.offsets import DateOffset
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from sklearn.preprocessing import MinMaxScaler

x_scaler = MinMaxScaler()
y_scaler = MinMaxScaler()

pred_len = 12
col_list = ['unix', 'symbol', 'open', 'high', 'low', 'close', 'Volume BTC', 'Volume USDT', 'tradecount','values','mindate']
df = pd.read_csv('./data/Binance_BTCUSDT_minute.csv', index_col='mindate', usecols=col_list,
                 low_memory=False, parse_dates=True)
df.drop(['symbol'], inplace=True, axis=1)
df = df.astype('float')

data_test = df.loc['2022-01-01':'2022-02-28 07:48']['values']
data_test = np.array(data_test)
data_test = data_test[::-1]

model = tf.keras.models.load_model('saved_model/MODEL2')
x_test, y_test = [], []
for i in range(pred_len, len(data_test)):
    x_test.append(data_test[i-pred_len:i])

print(pd.DataFrame(x_test))
x_test = np.array(x_test)
x_test = x_scaler.fit_transform(x_test)
x_test = x_test.reshape(len(x_test[:]), len(x_test[0]), 1)

# making test predictions to see how the models works
y_pred = model.predict(x_test)
y_pred = y_pred.reshape(len(x_test[:]), len(x_test[0]))
y_pred = x_scaler.inverse_transform(y_pred)
test_pred = y_pred[:, pred_len-1]
print(y_pred[:][-1])

# making future predictions by predicting a few times
future_pred = []
for i in range(0, 5*pred_len):
    print(i)
    y_pred = x_scaler.transform(y_pred)
    y_pred = model.predict(y_pred)
    y_pred = y_pred.reshape(len(x_test[:]), len(x_test[0]))
    y_pred = x_scaler.inverse_transform(y_pred)
    print(y_pred[-1, :])
    future_pred.append(y_pred[-1, pred_len-1])

values = df['2022-01-01':'2022-03-22'][['values']].astype(float)
print(len(future_pred))
print(len(test_pred))
# making array of test values
add_dates = [values.index[-pred_len]+DateOffset(minutes=x) for x in range(0, 5*len(test_pred)+1,5)]
dates = pd.DataFrame(index=add_dates[1:], columns=df.columns)
predict = pd.DataFrame(test_pred, index=dates.index, columns=['Test Prediction'])

# making array of future values
add_future_dates = [predict.index[-1]+DateOffset(minutes=x) for x in range(0, 5*len(future_pred)+1,5)]
future_dates = pd.DataFrame(index=add_future_dates[1:], columns=df.columns)
df_predict = pd.DataFrame(future_pred, index=future_dates.index, columns=['Future Prediction'])

values.rename(columns={'close': 'Real Values'}, inplace=True)
values['Test Prediction'] = predict
values['Future Prediction'] = df_predict
print(values)
values.rename(columns={'close': 'Real Values'}, inplace=True)
values.plot(figsize=(14, 5))
plt.show()