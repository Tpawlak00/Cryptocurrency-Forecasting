import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

x_scaler = MinMaxScaler()
y_scaler = MinMaxScaler()
pred_length = 12



def get_data(file):
    col_list = ['unix', 'symbol', 'open', 'high', 'low', 'close', 'Volume BTC', 'Volume USDT', 'tradecount', 'values',
                'mindate']
    df = pd.read_csv(file, index_col='mindate', usecols=col_list,
                     low_memory=False, parse_dates=True)
    df.drop(['symbol'], inplace=True, axis=1)
    df = df.astype('float')

    return df


def get_train_values():
    data = get_data('./data/Binance_BTCUSDT_minute.csv')
    train = data.loc['2020':'2021']['values']

    return train


def get_test_values():
    data = get_data('./data/Binance_BTCUSDT_minute2.csv')
    test = data.loc['2022-03-18':]['values']

    return test


def make_test_dataset():
    test = get_test_values()
    test = np.array(test[::-1])

    return test


def make_train_dataset():
    train = get_train_values()
    train = np.array(train[::-1])

    return train


def split_train_data():
    data_train = make_train_dataset()

    x_train, y_train = [], []
    for i in range(0, len(data_train) - 2*pred_length + 1, pred_length):
        x_train.append(data_train[i:i + pred_length])

    for i in range(pred_length, len(data_train) - pred_length + 1, pred_length):
        y_train.append(data_train[i:i + pred_length])

    print(pd.DataFrame(x_train))
    print(pd.DataFrame(y_train))
    x_train, y_train = np.array(x_train), np.array(y_train)
    print(x_train.shape, y_train.shape)

    return x_train, y_train


def split_test_data():
    data_test = make_test_dataset()

    x_test = []
    for i in range(0, len(data_test) - pred_length + 1, pred_length):
        x_test.append(data_test[i:i + pred_length])

    print(pd.DataFrame(x_test))
    x_test = np.array(x_test)
    print(x_test.shape)

    return x_test


def scale_train_data():
    inputs, labels = split_train_data()
    inputs = x_scaler.fit_transform(inputs)
    labels = y_scaler.fit_transform(labels)

    return inputs, labels


def scale_test_data():
    inputs = split_test_data()
    print(inputs)
    inputs = x_scaler.fit_transform(inputs)

    return inputs


def inverse_scaler(array):
    array = x_scaler.inverse_transform(array)

    return array
