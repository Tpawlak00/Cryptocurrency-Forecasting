import numpy as np
import pandas as pd
import tensorflow as tf

pred_length = 12


def read_file(file_path):
    col_list = ['price', 'date', 'target', 'AVG']
    df = pd.read_csv(file_path, index_col='date', usecols=col_list,
                     low_memory=False, parse_dates=True)
    return df


def one_hot(df, start_train, end_train, start_test):
    train_targets = df[start_train:end_train]
    test_targets = df[start_test:]

    targets_numpy = np.array(train_targets[::-1])
    test_targets_numpy = np.array(test_targets[::-1])

    to_encode = []
    for i in range(pred_length - 1, len(targets_numpy) - 1):
        if targets_numpy[i + 1] < targets_numpy[i]:
            to_encode.append([0])
        elif targets_numpy[i + 1] == targets_numpy[i]:
            to_encode.append([1])
        elif targets_numpy[i + 1] > targets_numpy[i]:
            to_encode.append([2])

    to_encode_test = []
    for i in range(pred_length - 1, len(test_targets_numpy) - 1):
        if test_targets_numpy[i + 1] < test_targets_numpy[i]:
            to_encode_test.append([0])
        elif test_targets_numpy[i + 1] == test_targets_numpy[i]:
            to_encode_test.append([1])
        elif test_targets_numpy[i + 1] > test_targets_numpy[i]:
            to_encode_test.append([2])

    encoded_val = tf.keras.utils.to_categorical(to_encode)
    encoded_test = tf.keras.utils.to_categorical(to_encode_test)

    return encoded_val, encoded_test


def split_data(df, start_train, end_train, start_test):
    train_data = df[start_train:end_train]
    test_data = df[start_test:]

    train_numpy = np.array(train_data[::-1])
    test_numpy = np.array(test_data[::-1])

    x_train = []
    for i in range(0, len(train_numpy) - pred_length):
        x_train.append(train_numpy[i:i + pred_length])

    x_test = []
    for i in range(0, len(test_numpy) - pred_length):
        x_test.append(test_numpy[i:i + pred_length])

    x_train = np.array(x_train)
    x_test = np.array(x_test)

    return x_train, x_test
