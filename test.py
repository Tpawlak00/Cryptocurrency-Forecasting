import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from get_dataset import read_file, one_hot, split_data

path = './data/BTC_minute.csv'
model_path = './saved_model/Conv.h5'
start_train, end_train = '3/16/2020  23:35:00', '3/01/2022  23:55:00'
start_test = '3/02/2022  00:00:00'
scaler = MinMaxScaler()


def make_prediction(model_path):
    model = tf.keras.models.load_model(model_path)

    data = read_file(path)
    _, inputs_test = split_data(data['price'], start_train, end_train, start_test)
    _, outputs_test = one_hot(data['target'], start_train, end_train, start_test)

    pred_list = []
    for i in range(0, len(inputs_test) - 2000):
        x_test = scaler.fit_transform(inputs_test[i:i + 2000])
        x_test = x_test[-1].reshape(1, -1)
        pred_list.append(model.predict(x_test))

    pred_list = np.array(pred_list).reshape(len(pred_list), 3)

    return pred_list


def filter(pred_array):
    for i in range(len(pred_array)):
        # get index of max value
        max_val = np.max(pred_array[i])
        # Find index position of maximum value using where()
        max_index = np.where(pred_array[i] == max_val)[0]
        sort_values = np.sort(pred_array[i])[::-1]

        # if max value is greater by 0.7 than the rest of values set categories
        if max_val - sort_values[1] > 0.7 and max_val - sort_values[2] > 0.7:
            if max_index == 0:
                pred_array[i] = [1, 0, 0]
            elif max_index == 1:
                pred_array[i] = [0, 1, 0]
            elif max_index == 2:
                pred_array[i] = [0, 0, 1]
        # else choose "hold" option
        else:
            pred_array[i] = [0, 1, 0]

    print(sum((pred_array == [1, 0, 0]).all(1)))
    print(sum((pred_array == [0, 1, 0]).all(1)))
    print(sum((pred_array == [0, 0, 1]).all(1)))
    return pred_array


def plot(path, test_data):
    data = read_file(path)
    data2 = data[::-1]
    plot_data = data2['price'][test_data:]
    plot_data = plot_data[2000:]
    sell_list = []
    buy_list = []
    print(len(pred))
    print(len(plot_data))
    for i in range(60, len(plot_data)):
        if np.all(pred[i-60] == [1, 0, 0]):
            sell_list.append(plot_data.index[i])
        if np.all(pred[i-60] == [0, 0, 1]):
            buy_list.append(plot_data.index[i])

    fig = plt.figure(figsize=(50, 25))
    ax1 = fig.add_subplot(111, ylabel='Price')

    plot_data.plot(ax=ax1, x_compat=True)
    ax1.plot(
        sell_list,
        plot_data[sell_list] + 5,
        'v', markersize=7, color='red'
    )

    ax1.plot(
        buy_list,
        plot_data[buy_list] - 5,
        '^', markersize=7, color='green'
    )
    plt.show()


if __name__ == "__main__":
    pred = filter(make_prediction(model_path))
    plot(path, start_test)
