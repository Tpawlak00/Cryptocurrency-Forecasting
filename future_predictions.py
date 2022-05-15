from pandas.tseries.offsets import DateOffset
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from main import scale_test_data, inverse_scaler, pred_length, get_data


def get_plot_values(filename):
    data = get_data(filename)
    plot_data = data['2022-03-18':][['values']].astype(float)

    return plot_data


def create_plot_data(filename, predictions, seq_length):
    values = get_plot_values(filename)

    add_dates = [values.index[-seq_length] + DateOffset(minutes=x) for x in range(0, 5 * len(predictions) + 1, 5)]
    dates = pd.DataFrame(index=add_dates[1:], columns=values.columns)
    predict = pd.DataFrame(predictions, index=dates.index, columns=['Test Prediction'])
    print(pd.DataFrame(predict))

    values.rename(columns={'close': 'Real Values'}, inplace=True)
    values['Test Prediction'] = predict

    return values


def plot(plot_data):
    print(pd.DataFrame(plot_data))
    plot_data.plot(figsize=(14, 5))
    plt.show()


def make_predictions():
    test = scale_test_data()
    test = test.reshape(len(test), len(test[0]), 1)

    model = tf.keras.models.load_model('saved_model/Model_5')
    y_pred = model.predict(test)
    y_pred = y_pred.reshape(len(test), len(test[0]))
    y_pred = inverse_scaler(y_pred)

    y_pred = np.array(y_pred)

    predictions_list = []
    for i in range(0, len(y_pred)):
        for j in range(0, len(y_pred[0])):
            predictions_list.append(y_pred[i, j])

    return predictions_list


plot(create_plot_data('./data/Binance_BTCUSDT_minute2.csv', make_predictions(), pred_length))
