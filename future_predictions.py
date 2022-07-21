import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from main import scale_data


def filter_data(pred_array):
  for i in range(len(pred_array)):
    #get index of max value
    max_val = np.max(pred_array[i])
    # Find index position of maximum value using where()
    max_index = np.where(pred_array[i] == max_val)[0]
    sort_values = np.sort(pred_array[i])[::-1]

    # if max value is greater by 0.2 than the rest of values set categories
    if max_val - sort_values[1] > 0.2 and max_val - sort_values[2] > 0.2:
      if max_index == 0:
        pred_array[i] = [1,0,0]
      elif max_index == 1:
        pred_array[i] = [0,1,0]
      elif max_index == 2:
        pred_array[i] = [0,0,1]
    # else choose "hold" option
    else:
      pred_array[i] = [0,1,0]


  return pred_array


def make_predictions(input_arr):
    model = tf.keras.models.load_model('saved_model/Conv_6_27_22')
    pred = model.predict(input_arr)
    pred = filter_data(pred)
    print(sum((pred == [1, 0, 0]).all(1)))
    print(sum((pred == [0, 1, 0]).all(1)))
    print(sum((pred == [0, 0, 1]).all(1)))
    return pred


def val_evaluate(pred_array, real_array):
    count = 0
    for i in range(len(pred_array)):
        if np.all(pred_array[i] == real_array[i]):
            count += 1
    score = count / len(real_array)
    return score


_, _, x_test, y_test = scale_data()
predictions = make_predictions(x_test)
print(val_evaluate(predictions, y_test))
