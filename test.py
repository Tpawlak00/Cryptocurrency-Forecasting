import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import tensorflow as tf
from get_dataset import scale_data, read_file
import xgboost as xgb

file_name = './data/BTC_tests.csv'
model_path = './saved_model/Conv.h5'
start_test = '2022-09-19 00:00:00'

def filter(pred_array):
    for i in range(len(pred_array)):
        # get index of max value
        max_val = np.max(pred_array[i])
        # Find index position of maximum value using where()
        max_index = np.where(pred_array[i] == max_val)[0]
        sort_values = np.sort(pred_array[i])[::-1]

        # if max value is greater by 0.7 than the rest of values set categories
        if max_val - sort_values[1] > 0.2 and max_val - sort_values[2] > 0.2:
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


def plot(path, pred_list):
    col_list = ['date', 'value', 'target']
    data = pd.read_csv(path, usecols=col_list,
                     low_memory=False, parse_dates=True)
    data = data[::-1]
    data = data.set_index('date')
    # data['date'] = pd.to_datetime(data['date'].dt.strftime('%m/%d/%Y'))
    plot_data = data['value']
    plot_data = plot_data[2016:]
    sell_list = []
    buy_list = []
    for i in range(12, len(plot_data)):
        if np.all(pred_list[i-12] == [1, 0, 0]):
            sell_list.append(plot_data.index[i])

        if np.all(pred_list[i-12] == [0, 0, 1]):
            buy_list.append(plot_data.index[i])
    print(sell_list)
    print(buy_list)
    fig = plt.figure(figsize=(12, 7))
    ax1 = fig.add_subplot(111, ylabel='Price')
    sell_list_index=[]
    buy_list_index=[]
    for i in range(len(sell_list)):
        sell_list_index.append(plot_data.index.get_loc(sell_list[i]))
    for i in range(len(buy_list)):
        buy_list_index.append(plot_data.index.get_loc(buy_list[i]))


    plot_data.plot(x=plot_data.index, ax=ax1, x_compat=True)
    ax1.plot(
        sell_list_index,
        plot_data[sell_list] + 5,
        'v', markersize=7, color='red'
      )

    ax1.plot(
        buy_list_index,
        plot_data[buy_list] - 5,
        '^', markersize=7, color='green'
    )
    plt.show()

def create_files():
    path = "./saved_model"
    inputs, _ = scale_data(file_name)
    for dirpath, dirnames, filenames in os.walk(path):
        try:
            dirname = dirpath.split(os.path.sep)[1]
        except IndexError:
            continue
        model = tf.keras.models.load_model(f'./saved_model/{dirname}/Conv.h5')
        pred = model.predict(np.array(inputs))
        data = np.array(pred)
        np.savetxt(f'./XBoost_data_test/{dirname}_predictions.csv', data, delimiter=',')


def concat_predictions():
    path = "./XBoost_data_test"
    filename = os.listdir(path)
    df1 = pd.read_csv(f'./XBoost_data_test/{filename[0]}')

    for i in range(1, len(filename)):
        print(f'Files done: {i}/{len(filename)}')
        df2 = pd.read_csv(f'./XBoost_data_test/{filename[i]}')
        df1 = pd.concat([df1, df2], axis=1)

    df1.to_csv('./XBoost_data_test/Superset_test.csv', index=False)

if __name__ == "__main__":
    # create_files()
    # concat_predictions()
    model_xgb_2 = xgb.Booster()
    model_xgb_2.load_model("model.txt")
    df = pd.read_csv('./XBoost_data_test/Superset_test.csv', sep=',', header=None, low_memory=False)
    inputs = df.to_numpy()
    test = xgb.DMatrix(inputs)
    pred = model_xgb_2.predict(test)
    # print(pd.DataFrame(X_test))
    print(pd.DataFrame(pred))
    print(pred.shape)
    pred = filter(pred)
    print(pred[0:100])
    plot(file_name, pred)
