import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf

pred_length = 12
scaler = MinMaxScaler()


def read_file(file_path):
    col_list = ['date', 'open', 'close', 'high', 'low', 'target']
    df = pd.read_csv(file_path, index_col='date', usecols=col_list,
                     low_memory=False, parse_dates=True)

    targets_numpy = np.array(df['target'][::-1])

    to_encode = []
    for i in range(0, len(targets_numpy) - 1):
        if targets_numpy[i + 1] < targets_numpy[i]:
            to_encode.append(0)
        elif targets_numpy[i + 1] == targets_numpy[i]:
            to_encode.append(1)
        elif targets_numpy[i + 1] > targets_numpy[i]:
            to_encode.append(2)

    to_encode.append(1)
    df = df[::-1]
    df['target'] = to_encode

    return df


def split_data(filepath):
    df = read_file(filepath)
    train_data_open, train_data_close, train_low, train_high, labels = df['open'], df['close'], df['low'],df['high'], df['target']

    train_numpy_open = np.array(train_data_open)
    train_numpy_close = np.array(train_data_close)
    train_numpy_low = np.array(train_low)
    train_numpy_high = np.array(train_high)
    labels_numpy = np.array(labels[pred_length:])

    x_train_open = []
    for i in range(0, len(train_numpy_open) - pred_length+1):
        x_train_open.append(train_numpy_open[i:i + pred_length])

    x_train_close = []
    for i in range(0, len(train_numpy_close) - pred_length+1):
        x_train_close.append(train_numpy_close[i:i + pred_length])

    x_train_low = []
    for i in range(0, len(train_numpy_low) - pred_length+1):
        x_train_low.append(train_numpy_low[i:i + pred_length])

    x_train_high = []
    for i in range(0, len(train_numpy_high) - pred_length+1):
        x_train_high.append(train_numpy_high[i:i + pred_length])

    x_train_open = np.array(x_train_open)
    x_train_close = np.array(x_train_close)
    x_train_low = np.array(x_train_low)
    x_train_high = np.array(x_train_high)

    return x_train_open, x_train_close, x_train_low, x_train_high, labels_numpy


def scale_data(filepath):
    x_train_open, x_train_close, x_train_low, x_train_high, y_train = split_data(filepath)
    y_train = y_train[2003:]
    y_train = tf.keras.utils.to_categorical(y_train)

    x_train_scaled_open = np.array([scaler.fit_transform(x_train_open[i:i+2005])[-1] for i in range(len(x_train_open)-2004)])
    x_train_scaled_close = np.array([scaler.fit_transform(x_train_close[i:i+2005])[-1] for i in range(len(x_train_close)-2004)])
    x_train_scaled_low = np.array([scaler.fit_transform(x_train_low[i:i+2005])[-1] for i in range(len(x_train_low)-2004)])
    x_train_scaled_high = np.array([scaler.fit_transform(x_train_high[i:i+2005])[-1] for i in range(len(x_train_high)-2004)])

    x_train = np.stack(([x_train_scaled_open, x_train_scaled_close, x_train_scaled_high, x_train_scaled_low]), axis=1)


    return np.array(x_train), np.array(y_train)

# scale_data("./data/BTC_WMA_2020.csv")