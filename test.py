import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import pandas as pd
import os
import tensorflow as tf
from get_dataset import scale_data, read_file
import xgboost as xgb
from sklearn.metrics import accuracy_score
import pickle
from datetime import datetime, timedelta

file_name = './data/BTC_tests3.csv'
model_path = './saved_model/Conv.h5'
start_test = '2022-09-19 00:00:00'


def create_files():
    path = "./saved_model"
    x_train, _ = scale_data(file_name)
    for dirpath, dirnames, filenames in os.walk(path):
        try:
            dirname = dirpath.split(os.path.sep)[1]
        except IndexError:
            continue
        model = tf.keras.models.load_model(f'./saved_model/{dirname}/Conv.h5')
        predictions = model.predict(np.array(x_train))
        file_data = np.array(predictions)
        np.savetxt(f'./XBoost_data_test/{dirname}_predictions.csv', file_data, delimiter=',')


def concat_predictions():
    path = "./XBoost_data_test"
    filename = os.listdir(path)
    df1 = pd.read_csv(f'./XBoost_data_test/{filename[0]}')

    for i in range(1, len(filename)):
        print(f'Files done: {i}/{len(filename)}')
        df2 = pd.read_csv(f'./XBoost_data_test/{filename[i]}')
        df1 = pd.concat([df1, df2], axis=1)

    df1.to_csv('./XBoost_data_test/Superset_test.csv', index=False)


def calculate_earnings(dataframe, outputs, predictions, buy_prob, sell_prob):
    df_check = pd.DataFrame()
    df_check['value'], df_check['date'] = dataframe['value'][2027:], dataframe['date'][2027:]
    df_check.index = df_check['date']
    df_check = df_check.drop(['date'], axis=1)

    action_list = []
    for i in range(len(outputs)):
        # get index of max value
        max_val = np.max(outputs[i])
        # Find index position of maximum value using where()
        max_index = np.where(outputs[i] == max_val)[0]
        if max_index == 0:
            action_list.append("S")
        elif max_index == 1:
            action_list.append("H")
        elif max_index == 2:
            action_list.append("B")
    df_check["action"] = action_list

    capital = 1
    value = 0
    earnings_list = [1]
    for i in range(1, len(df_check)):
        if df_check["action"][i] == "B":
            value = capital / df_check["value"][i]
        elif df_check["action"][i] == "H":
            value = value
            capital = capital
        elif df_check["action"][i] == "S":
            capital = value * df_check["value"][i]
        earnings_list.append(capital)
    df_check["estimated earnings"] = earnings_list

    bot_actions = []
    # calculate bot earnings
    for i in range(len(predictions)):
        max_bot_val = np.max(predictions[i])
        # Find index position of maximum value using where()
        max_bot_index = np.where(predictions[i] == max_bot_val)[0]
        if max_bot_index == 0 and max_bot_val > buy_prob:
            bot_actions.append("S")
        elif max_bot_index == 2 and max_bot_val > sell_prob:
            bot_actions.append("B")
        else:
            bot_actions.append("H")
    df_check["bot action"] = bot_actions

    for i in range(len(df_check)):
        if df_check["bot action"][i] == 'S':
            df_check["bot action"][i] = 'H'
        elif df_check["bot action"][i] == 'H':
            continue
        elif df_check["bot action"][i] == 'B':
            break

    bot_capital = 1
    bot_value = 0
    bot_earnings = []
    for i in range(0, len(df_check)):
        if df_check["bot action"][i] == "B":
            if bot_capital == 0:
                earnings = bot_value * df_check["value"][i]
                bot_earnings.append(earnings)
                continue
            bot_value = bot_capital / df_check["value"][i]
            bot_earnings.append(bot_capital)
            bot_capital = 0
            continue
        elif df_check["bot action"][i] == "H":
            bot_value = bot_value
            bot_capital = bot_capital
            if bot_capital == 0:
                earnings = bot_value * df_check["value"][i]
                bot_earnings.append(earnings)
                continue
            elif bot_value == 0:
                bot_earnings.append(bot_capital)
                continue
        elif df_check["bot action"][i] == "S":
            if bot_value == 0:
                bot_earnings.append(bot_capital)
                continue
            bot_capital = bot_value * df_check["value"][i]
            bot_value = 0
            bot_earnings.append(bot_capital)
            continue

    df_check["bot earnings"] = bot_earnings

    accuracy = df_check["bot earnings"][-1] / df_check["estimated earnings"][-1]
    # d1 = datetime.strptime(df_check.index[0], '%Y-%m-%d %H:%M:%S')
    # d2 = datetime.strptime(df_check.index[-1], '%Y-%m-%d %H:%M:%S')
    # d1 = datetime.strptime(df_check.index[0], '%d.%m.%Y %H:%M')
    # d2 = datetime.strptime(df_check.index[-1], '%d.%m.%Y %H:%M')
    # delta = d2 - d1
    # print("First day:", df_check.index[0], "last day:", df_check.index[-1])
    # print("Earned:", bot_earnings[-1], "of starting capital in:", delta.days, "days")
    # print("Estimated capital income:", df_check["estimated earnings"][-1])
    # print("Accuracy: %.2f%%" % (accuracy * 100.0))

    # df_check.plot(y=["bot earnings"])
    # plt.show()  # Depending on whether you use IPython or interactive mode, etc.

    return accuracy, df_check["bot earnings"]


def find_best_prob(x_pred):
    acc_list = []
    i_list = []
    j_list = []
    for i in range(30, 100):
        print(i)
        for j in range(30, 100):
            to_add, _ = calculate_earnings(df, test2y, x_pred, i/100, j/100)
            acc_list.append(to_add)
            i_list.append(i)
            j_list.append(j)

    max_value = max(acc_list)
    max_index = acc_list.index(max_value)
    print(f"Accuracy: {max_value*100}% for buy probability = 0,{i_list[max_index]} and sell"
          f" probability = 0,{j_list[max_index]}")

    return max_index*100, i_list[max_index], j_list[max_index]


def plot_bar_prob(df_bar, pred):
    df_bar['Buy'], df_bar['Hold'], df_bar['Sell'] = np.array(pred[:, 2]), pred[:, 1], pred[:, 0]
    df_bar[1200:1300].plot.bar(y=["Buy", "Sell", "Hold"], rot=45)
    plt.axhline(y=0.5, color='black', linestyle = "--")
    plt.show()

    return df_bar

def filter(pred_array):
    for i in range(len(pred_array)):
        # get index of max value
        max_val = np.max(pred_array[i])
        # Find index position of maximum value using where()
        max_index = np.where(pred_array[i] == max_val)[0]
        sort_values = np.sort(pred_array[i])[::-1]

        # if max value is greater by 0.7 than the rest of values set categories
        if max_val > 0.45:
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

def plot_decisions(path, pred_list):
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

def plot(path, pred_list):
    col_list = ['date', 'value', 'target']
    data = pd.read_csv(path, usecols=col_list,
                     low_memory=False, parse_dates=True)
    data = data[::-1]
    data = data.set_index('date')
    # data['date'] = pd.to_datetime(data['date'].dt.strftime('%m/%d/%Y'))
    plot_data = data['value']
    plot_data = plot_data[2015:]
    plot_data = plot_data[1:289]
    pred_list = pred_list[1:289]
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
    ax1 = fig.add_subplot(111, ylabel='Cena [$]', xlabel='Data [GG:MM]')

    plot_data.index = pd.to_datetime(plot_data.index).strftime('%H:%M')
    sell_list = pd.to_datetime(sell_list).strftime('%H:%M')
    buy_list = pd.to_datetime(buy_list).strftime('%H:%M')

    sell_index=[]
    for i in range(len(sell_list)):
        sell_index.append(plot_data.index.get_loc(sell_list[i]))

    buy_index=[]
    for i in range(len(buy_list)):
        buy_index.append(plot_data.index.get_loc(buy_list[i]))

    print(sell_list)
    print(buy_list)
    print(plot_data.index)

    plot_data.plot(x=plot_data.index,xlabel='Godzina [GG:MM]', ax=ax1, x_compat=True, rot=45,
                   grid=True, title='Oznaczenia kupna i sprzedaży bitcoina w dniu 26-09-2022')
    ax1.plot(
        sell_index,
        plot_data[sell_list] + 5,
        'v', markersize=7, color='red'
      )

    ax1.plot(
        buy_index,
        plot_data[buy_list] - 5,
        '^', markersize=7, color='green'
    )
    plt.legend(["Cena", "Sprzedaż", "Kupno"])
    plt.show()

if __name__ == "__main__":
    # create_files()
    # concat_predictions()
    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
    df = pd.read_csv('./XBoost_data_test/Superset_test.csv', sep=',', header=None, low_memory=False)


    inputs = df.to_numpy()

    x_train, _ = scale_data(file_name)
    # model1 = tf.keras.models.load_model(f'./saved_model/From 01-01-2021 00-00 To 01-31-2021 00-00/Conv.h5')
    # model2 = tf.keras.models.load_model(f'./saved_model/From 01-31-2021 00-00 To 03-02-2021 00-00/Conv.h5')
    # model3 = tf.keras.models.load_model("./saved_model/From 03-02-2021 00-00 To 04-01-2021 00-00/Conv.h5")
    # model4 = tf.keras.models.load_model(f'./saved_model/From 04-01-2021 00-00 To 05-01-2021 00-00/Conv.h5')
    # model5 = tf.keras.models.load_model(f'./saved_model/From 05-01-2021 00-00 To 05-31-2021 00-00/Conv.h5')
    model6 = tf.keras.models.load_model(f'./saved_model/From 05-31-2021 00-00 To 06-30-2021 00-00/Conv.h5')
    # model7 = tf.keras.models.load_model(f'./saved_model/From 06-30-2021 00-00 To 07-30-2021 00-00/Conv.h5')
    # model8 = tf.keras.models.load_model(f'./saved_model/From 07-30-2021 00-00 To 08-29-2021 00-00/Conv.h5')
    # model9 = tf.keras.models.load_model(f'./saved_model/From 08-29-2021 00-00 To 09-28-2021 00-00/Conv.h5')
    # model10 = tf.keras.models.load_model(f'./saved_model/From 09-28-2021 00-00 To 10-28-2021 00-00/Conv.h5')
    # model11 = tf.keras.models.load_model(f'./saved_model/From 10-28-2021 00-00 To 11-27-2021 00-00/Conv.h5')
    # model12 = tf.keras.models.load_model(f'./saved_model/From 11-27-2021 00-00 To 12-27-2021 00-00/Conv.h5')
    # model13 = tf.keras.models.load_model(f'./saved_model/From 3-16-2020 23-35 To 12-31-2020 23-55/Conv.h5')

    # model1_pred = model1.predict(np.array(x_train))
    # model2_pred = model2.predict(np.array(x_train))
    # model3_pred = model3.predict(np.array(x_train))
    # model4_pred = model4.predict(np.array(x_train))
    # model5_pred = model5.predict(np.array(x_train))
    model6_pred = model6.predict(np.array(x_train))
    # model7_pred = model7.predict(np.array(x_train))
    # model8_pred = model8.predict(np.array(x_train))
    # model9_pred = model9.predict(np.array(x_train))
    # model10_pred = model10.predict(np.array(x_train))
    # model11_pred = model11.predict(np.array(x_train))
    # model12_pred = model12.predict(np.array(x_train))
    # model13_pred = model13.predict(np.array(x_train))

    model_xgb_2 = xgb.Booster()
    model_xgb_2.load_model("xgb_12model2.txt")
    test = xgb.DMatrix(inputs)
    xgb_pred = model_xgb_2.predict(test)


    ada_model = pickle.load(open("ada_12model10.sav", 'rb'))
    ada_pred = ada_model.predict_proba(inputs)

    _, test2y = scale_data(file_name)

    col_list = ['date', 'value', 'target']
    df = pd.read_csv(file_name,
                     usecols=col_list,
                     low_memory=False, parse_dates=True)
    df = df[::-1]

    data = pd.DataFrame()
    data['Value'], data['date'], data['target'] = df['value'][2027:], df['date'][2027:], df['target'][2027:]
    data.set_index('date')
    data.index = data['date']

    _, data["Kapitał sieci neuronowej"] = calculate_earnings(df, test2y, model6_pred, 0.76, 0.68)  # xgb
    _, data["Kapitał XGBoost"] = calculate_earnings(df, test2y, xgb_pred, 0.49, 0.49) # xgb
    _, data["Kapitał ADABoost"] = calculate_earnings(df, test2y, ada_pred, 0.49, 0.61)  # ada

    data.index = pd.to_datetime(data.index).strftime("%d-%m")

    data.plot(y=["Kapitał XGBoost", "Kapitał sieci neuronowej", "Kapitał ADABoost"], rot=45, grid=True,
              ylabel="Wartość kapitału", xlabel="Data [DD-MM]", title="Wartość kapitału modeli w zależności od"
                                                                      " czasu w dniach od 26-09-2022 do 07-11-2022",
              legend=True, x_compat=True)
    plt.show()  # Depending on whether you use IPython or interactive mode, etc.

    # plot_conv = plot_bar_prob(data, model3_pred)
    # plot_xgb = plot_bar_prob(data, xgb_pred)
    # plot_ada = plot_bar_prob(data, ada_pred)

    # acc_conv, buy_conv, sell_conv = find_best_prob(model3_pred)
    # acc_xgb, buy_xgb, sell_xgb = find_best_prob(xgb_pred)
    # acc_ada, buy_ada, sell_ada = find_best_prob(ada_pred)

    # acc_conv1, buy_conv1, sell_conv1 = find_best_prob(model1_pred)
    # acc_conv2, buy_conv2, sell_conv2 = find_best_prob(model2_pred)
    # acc_conv3, buy_conv3, sell_conv3 = find_best_prob(model3_pred)
    # acc_conv4, buy_conv4, sell_conv4 = find_best_prob(model4_pred)
    # acc_conv5, buy_conv5, sell_conv5 = find_best_prob(model5_pred)
    # acc_conv6, buy_conv6, sell_conv6 = find_best_prob(model6_pred)
    # acc_conv7, buy_conv7, sell_conv7 = find_best_prob(model7_pred)
    # acc_conv8, buy_conv8, sell_conv8 = find_best_prob(model8_pred)
    # acc_conv9, buy_conv9, sell_conv9 = find_best_prob(model9_pred)
    # acc_conv10, buy_conv10, sell_conv10 = find_best_prob(model10_pred)
    # acc_conv11, buy_conv11, sell_conv11 = find_best_prob(model11_pred)
    # acc_conv12, buy_conv12, sell_conv12 = find_best_prob(model12_pred)
    # acc_conv13, buy_conv13, sell_conv13 = find_best_prob(model13_pred)

    # print("for model1 buy:",buy_conv1,"sell:",sell_conv1)
    # print("for model2 buy:", buy_conv2, "sell:", sell_conv2)
    # print("for model3 buy:", buy_conv3, "sell:", sell_conv3)
    # print("for model4 buy:", buy_conv4, "sell:", sell_conv4)
    # print("for model5 buy:", buy_conv5, "sell:", sell_conv5)
    # print("for model6 buy:", buy_conv6, "sell:", sell_conv6)
    # print("for model7 buy:", buy_conv7, "sell:", sell_conv7)
    # print("for model8 buy:", buy_conv8, "sell:", sell_conv8)
    # print("for model9 buy:", buy_conv9, "sell:", sell_conv9)
    # print("for model10 buy:", buy_conv10, "sell:", sell_conv10)
    # print("for model11 buy:", buy_conv11, "sell:", sell_conv11)
    # print("for model12 buy:", buy_conv12, "sell:", sell_conv12)
    # print("for model13 buy:", buy_conv13, "sell:", sell_conv13)




    # print("conv:",buy_conv,sell_conv)
    # print("xgb:",buy_xgb,sell_xgb)
    # print("ada:",buy_ada,sell_ada)

    # _, data["Conv model 1 earnings"] = calculate_earnings(df, test2y, model1_pred, 0.73, 0.97)  # xgb
    # _, data["Conv model 2 earnings"] = calculate_earnings(df, test2y, model2_pred, 0.76, 0.52)  # xgb
    # _, data["Conv model 3 earnings"] = calculate_earnings(df, test2y, model3_pred, 0.77, 0.75)  # xgb
    # _, data["Conv model 4 earnings"] = calculate_earnings(df, test2y, model4_pred, 0.69, 0.48)  # xgb
    # _, data["Conv model 5 earnings"] = calculate_earnings(df, test2y, model5_pred, 0.61, 0.93)  # xgb
    # _, data["Conv model 6 earnings"] = calculate_earnings(df, test2y, model6_pred, 0.76, 0.68)  # xgb
    # _, data["Conv model 7 earnings"] = calculate_earnings(df, test2y, model7_pred, 0.80, 0.83)  # xgb
    # _, data["Conv model 8 earnings"] = calculate_earnings(df, test2y, model8_pred, 0.62, 0.58)  # xgb
    # _, data["Conv model 9 earnings"] = calculate_earnings(df, test2y, model9_pred, 0.85, 0.93)  # xgb
    # _, data["Conv model 10 earnings"] = calculate_earnings(df, test2y, model10_pred, 0.74, 0.98)  # xgb
    # _, data["Conv model 11 earnings"] = calculate_earnings(df, test2y, model11_pred, 0.55, 0.69)  # xgb
    # _, data["Conv model 12 earnings"] = calculate_earnings(df, test2y, model12_pred, 0.92, 0.93)  # xgb
    # _, data["Conv model 13 earnings"] = calculate_earnings(df, test2y, model13_pred, 0.30, 0.51)  # xgb

    # data.plot(y=["Conv model 1 earnings", "Conv model 2 earnings", "Conv model 3 earnings", "Conv model 4 earnings",
    #              "Conv model 5 earnings", "Conv model 6 earnings", "Conv model 7 earnings", "Conv model 8 earnings",
    #              "Conv model 9 earnings", "Conv model 10 earnings", "Conv model 11 earnings", "Conv model 12 earnings",
    #              "Conv model 13 earnings"], rot=45)
    # plt.show()  # Depending on whether you use IPython or interactive mode, etc.

    # plot_decisions(plot_conv, 0.54, 0.84)
    # plot_decisions(plot_xgb, 0.41, 0.37)
    # ada_pred = filter(ada_pred)
    # plot_decisions('./data/BTC_tests3.csv', ada_pred)

    # plot(file_name, test2y)
