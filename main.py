import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder

scaler = MinMaxScaler()
pred_length = 12  # 5 days
path = './data/BTC_minute.csv'
encoder = OneHotEncoder()


def make_dataset(file):
    col_list = ['price', 'date', 'target']
    df = pd.read_csv(file, index_col='date', usecols=col_list,
                     low_memory=False, parse_dates=True)
    df['price'] = df['price'].astype('float')
    df['target'] = df['target'].astype('int')

    train_targets = df['target']

    y_train = ['Hold']

    for i in range(0, len(train_targets) - 1):
        if train_targets[i] < train_targets[i + 1]:
            y_train.append('Sell')
        elif train_targets[i] == train_targets[i + 1]:
            y_train.append('Hold')
        elif train_targets[i] > train_targets[i + 1]:
            y_train.append('Buy')

    df['target'] = y_train

    encoder_df = pd.DataFrame(encoder.fit_transform(df[['target']]).toarray(), index=df.index)

    return df, encoder_df


def split_data():
    inputs, outputs = make_dataset(path)
    x_train, y_train = inputs['3/16/2020  23:35:00':'12/31/2021  23:55:00']['price'], outputs[
                                                                                       '3/16/2020  23:35:00':'12/31/2021  23:55:00']
    x_test, y_test = inputs['2022']['price'], outputs['2022']

    x_train = np.array(x_train[::-1])

    y_train = np.array(y_train)
    y_train = np.flip(y_train)
    y_train = y_train[pred_length - 1:]

    x_test = np.array(x_test[::-1])

    y_test = np.array(y_test)
    y_test = np.flip(y_test)
    y_test = y_test[pred_length - 1:]

    x_train_out, x_test_out = [], []
    for i in range(0, len(x_train) - pred_length + 1):
        x_train_out.append(x_train[i:i + pred_length])

    for i in range(0, len(x_test) - pred_length + 1):
        x_test_out.append(x_test[i:i + pred_length])

    x_train_out, x_test_out = np.array(x_train_out), np.array(x_test_out)

    return x_train_out, y_train, x_test_out, y_test


def scale_data():
    inputs_train, outputs_train, inputs_test, outputs_test = split_data()
    inputs_train = scaler.fit_transform(inputs_train)
    inputs_test = scaler.transform(inputs_test)

    inputs_train = inputs_train.reshape(len(inputs_train), pred_length, 1)
    inputs_test = inputs_test.reshape(len(inputs_test), pred_length, 1)

    return inputs_train, outputs_train, inputs_test, outputs_test
