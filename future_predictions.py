import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from main import scale_test_data, get_test_values, pred_length

#Na razie tylko poglądowo, potem zrobię oznaczanie na osi czasu

def filter(array):
    for i in range(len(array)):
        if array[i,0] > 0.6:
            array[i] = [1,0,0]
        elif array[i,1] > 0.6:
            array[i] = [0,1,0]
        elif array[i,2] > 0.6:
            array[i] = [0,0,1]

    return array

def plot():
    test = get_test_values()
    test.plot(figsize=(14, 5))
    plt.show()

if __name__ == '__main__':
    test = scale_test_data()
    print(pd.DataFrame(test))
    test = test.reshape(len(test), len(test[0]), 1)
    model = tf.keras.models.load_model('saved_model/Conv_6_18_22')
    pred = model.predict(test)
    print(pred[0:20])
    pred = filter(pred)
    print(pred[0:20])
    plot()
