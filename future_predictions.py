import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from main import scale_data


def filter(array):
    for i in range(len(array)):
        if array[i,0] > 0.6:
            array[i] = [1,0,0]
        elif array[i,1] > 0.6:
            array[i] = [0,1,0]
        elif array[i,2] > 0.6:
            array[i] = [0,0,1]
        else:
          array[i] = [0,1,0]

    return array



_, _, x_test, y_test = scale_data()

model = tf.keras.models.load_model('saved_model/Conv_6_22_22')
pred = model.predict(x_test)
print(pred[0:20])
pred = filter(pred)
print(pred[0:20])

def val_evaluate(pred_array, real_array):
  count = 0
  for i in range(len(pred_array)):
    if np.all(pred_array[i] == real_array[i]):
      count += 1
  score = count/len(real_array)
  return score

val_evaluate(pred, y_test)

