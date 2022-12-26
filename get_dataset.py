import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf

pred_length = 12
scaler = MinMaxScaler()


def read_file(file_path):
    col_list = ['date', 'value', 'target']
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
    train_data, labels = df['value'], df['target']

    train_numpy = np.array(train_data)
    labels_numpy = np.array(labels[pred_length:])

    x_train = []
    for i in range(0, len(train_numpy) - pred_length+1):
        x_train.append(train_numpy[i:i + pred_length])

    x_train = np.array(x_train)

    return x_train, labels_numpy


def scale_data(filepath):
    x_train, y_train = split_data(filepath)
    x_train_scaled = []
    y_train = y_train[2003:]
    y_train = tf.keras.utils.to_categorical(y_train)

    for i in range(len(x_train)-2004):
        scaled_df = scaler.fit_transform(x_train[0 + i:2005 + i])
        x_train_scaled.append(scaled_df[-1])

    return np.array(x_train_scaled), np.array(y_train)