import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
pred_length = 1440  # 5 days
path = './data/BTC_minute.csv'

def get_data(file):
    col_list = ['price', 'date', 'target']
    df = pd.read_csv(file, index_col='date', usecols=col_list,
                     low_memory=False, parse_dates=True)
    df['price'] = df['price'].astype('float')
    df['target'] = df['target'].astype('int')
    print(df)

    return df


def get_train_values():
    data = get_data(path)
    train_labels = data.loc['2020-03-16 23:35':'2021-12-31 23:55']['price']
    train_targets = data.loc['2020-03-16 23:35':'2021-12-31 23:55']['target']

    return train_labels, train_targets


def get_test_values():
    data = get_data(path)
    test = data.loc['2022']['price']

    return test


def make_test_dataset():
    test = get_test_values()
    test = np.array(test[::-1])

    return test


def make_train_dataset():
    train_inputs, train_targets = get_train_values()
    train_inputs = np.array(train_inputs[::-1])
    train_targets = np.array(train_targets[::-1])

    y_train = []
    y_train.append([0, 1, 0])

    for i in range(0, len(train_targets)-1):
        if train_targets[i] < train_targets[i + 1]:
            y_train.append([1,0,0])
        elif train_targets[i] == train_targets[i + 1]:
            y_train.append([0,1,0])
        elif train_targets[i] > train_targets[i + 1]:
            y_train.append([0,0,1])

    return train_inputs, y_train


def split_train_data():
    inputs, targets = make_train_dataset()

    x_train, y_train = [], []
    for i in range(0, len(inputs) - pred_length):
        x_train.append(inputs[i:i + pred_length])

    for i in range(pred_length, len(targets)):
        y_train.append(targets[i])

    x_train, y_train = np.array(x_train), np.array(y_train)

    print('Inputs shape:', x_train.shape, 'targets shape:', y_train.shape)

    return x_train, y_train


def split_test_data():
    data_test = make_test_dataset()

    x_test = []
    for i in range(0, len(data_test) - pred_length):
        x_test.append(data_test[i:i + pred_length])

    print(pd.DataFrame(x_test))
    x_test = np.array(x_test)
    print(x_test.shape)

    return x_test


def scale_train_data():
    inputs, labels = split_train_data()
    inputs = scaler.fit_transform(inputs)
    print(pd.DataFrame(inputs))
    print(pd.DataFrame(labels))
    return inputs, labels


def scale_test_data():
    _, _ = scale_train_data()
    inputs = split_test_data()
    inputs = scaler.transform(inputs)

    return inputs
