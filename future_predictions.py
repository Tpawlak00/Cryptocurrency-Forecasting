# from pandas.tseries.offsets import DateOffset
# import pandas as pd
# import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from main import scale_test_data, inverse_scaler, pred_length, get_data, scale_train_data


# def get_plot_values(filename):
#     data = get_data(filename)
#     plot_data = data['2022-03-18':][['values']].astype(float)
#
#     return plot_data
#
#
# def create_plot_data(filename, predictions, seq_length):
#     values = get_plot_values(filename)
#
#     add_dates = [values.index[-seq_length] + DateOffset(minutes=x) for x in range(0, 5 * len(predictions) + 1, 5)]
#     dates = pd.DataFrame(index=add_dates[1:], columns=values.columns)
#     predict = pd.DataFrame(predictions, index=dates.index, columns=['Test Prediction'])
#     print(pd.DataFrame(predict))
#
#     values.rename(columns={'close': 'Real Values'}, inplace=True)
#     values['Test Prediction'] = predict
#
#     return values
#
#
# def plot(plot_data):
#     print(pd.DataFrame(plot_data))
#     plot_data.plot(figsize=(14, 5))
#     plt.show()


def make_predictions():
    test = scale_test_data()
    test = test.reshape(len(test), len(test[0]), 1)
    model = tf.keras.models.load_model('saved_model/Model_11')

    scaled_predictions, predictions_list = [], []

    for i in range(0, pred_length):
        y_pred = model.predict(test)
        scaled_predictions.append(y_pred[0, -1][0])
        test = np.concatenate((test[0], [y_pred[0, -1]]), axis=0)
        test = np.delete(test, 0, axis=0)
        test = test.reshape(1, len(test), 1)

    print(scaled_predictions)
    predictions_list = inverse_scaler(np.array(scaled_predictions).reshape(1,-1)).flatten().tolist()
    print(predictions_list)

    return scaled_predictions, predictions_list

make_predictions()
# plot(create_plot_data('./data/Binance_BTCUSDT_minute2.csv', make_predictions(), pred_length))