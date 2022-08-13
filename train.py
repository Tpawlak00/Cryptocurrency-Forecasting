import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import matplotlib.pyplot as plt
from get_dataset import read_file, one_hot, split_data, pred_length
from datetime import datetime, timedelta
import os

scaler = MinMaxScaler()
file_path = './data/BTC_minute.csv'
start_train, end_train = '3/16/2020  23:35:00', '3/01/2022  23:55:00'
start_test = '3/02/2022  00:00:00'


def build_model():
    model_name = tf.keras.models.Sequential()

    model_name.add(tf.keras.layers.Conv1D(32, 2, input_shape=(pred_length, 1)))
    # model.add(tf.keras.layers.AveragePooling1D(2))
    # model_name.add(tf.keras.layers.LSTM(16))
    model_name.add(tf.keras.layers.Flatten())

    model_name.add(tf.keras.layers.Dense(16, activation='relu'))
    # model.add(tf.keras.layers.Dropout(0.2))
    model_name.add(tf.keras.layers.Dense(16, activation='relu'))
    model_name.add(tf.keras.layers.BatchNormalization())
    # model.add(tf.keras.layers.Dropout(0.2))
    model_name.add(tf.keras.layers.Dense(64, activation='relu'))
    model_name.add(tf.keras.layers.Dense(32, activation='relu'))
    model_name.add(tf.keras.layers.BatchNormalization())
    # model.add(tf.keras.layers.Dropout(0.5))

    model_name.add(tf.keras.layers.Dense(3, activation='softmax'))

    model_name.compile(loss=tf.keras.losses.CategoricalCrossentropy(),
                       optimizer=tf.keras.optimizers.Adam(),
                       metrics=['accuracy'])
    model_name.summary()

    return model_name


def make_folder(date1, date2):
    date_path1 = date1.strftime('%m-%d-%Y %H-%M')
    date_path2 = date2.strftime('%m-%d-%Y %H-%M')
    folder_path = f'./saved_model/From {date_path1} To {date_path2}'
    os.mkdir(folder_path)
    return folder_path


def scale_data(input_df, output_df):
    scaled = []
    for i in range(0, len(input_df) - 2016):
        get_rows = scaler.fit_transform(input_df[i:2016 + i])
        scaled.append((get_rows[-1]))
    scaled = np.array(scaled)
    output_df = output_df[2016:]

    return scaled, output_df


def model_checkpoint(model_path):
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=f'{model_path}/Conv.h5',
        save_weights_only=False,
        monitor='accuracy',
        mode='max',
        save_best_only=True,
        verbose=0)
    return model_checkpoint_callback


if __name__ == "__main__":
    start_new = datetime(2021, 1, 1, 0, 0)
    end_new = datetime(2021, 1, 8, 0, 0)

    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
    path = f'./saved_model/From 3-16-2020 23-35 To 12-31-2020 23-55'
    os.mkdir(path)

    data = read_file(file_path)
    inputs_train, _ = split_data(data['price'], start_train, end_train, start_test)
    outputs_train, _ = one_hot(data['target'], start_train, end_train, start_test)
    scaled_x, scaled_y = scale_data(inputs_train, outputs_train)

    # 83080 samples between 16-03-2020 31-12-2020
    x_train, y_train = scaled_x[0:83080], scaled_y[0:83080]
    model = build_model()
    history = model.fit(x_train, y_train, epochs=1000, batch_size=16,
                        validation_split=0.2,
                        verbose=1,
                        callbacks=[model_checkpoint(path)])

    plt.plot(history.history['accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train'], loc='upper left')
    plt.savefig(f'{path}/accuracy')
    plt.clf()

    plt.plot(history.history['loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train'], loc='upper left')
    plt.savefig(f'{path}/loss')
    plt.clf()

    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train'], loc='upper left')
    plt.savefig(f'{path}/val_acc')
    plt.clf()

    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train'], loc='upper left')
    plt.savefig(f'{path}/val_loss')
    plt.clf()

    x_train, y_train = scaled_x[83080:], scaled_y[83080:]

    for i in range(0, len(x_train), 12):
        model = tf.keras.models.load_model(f'{path}/Conv.h5')
        path = make_folder(start_new, end_new)
        history = model.fit(x_train[i:2016+i], y_train[i:2016+i], epochs=100, batch_size=16,
                            validation_split=0.2,
                            verbose=1,
                            callbacks=[model_checkpoint(path)])

        plt.plot(history.history['accuracy'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train'], loc='upper left')
        plt.savefig(f'{path}/accuracy')
        plt.clf()

        plt.plot(history.history['loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train'], loc='upper left')
        plt.savefig(f'{path}/loss')
        plt.clf()

        plt.plot(history.history['val_accuracy'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train'], loc='upper left')
        plt.savefig(f'{path}/val_acc')
        plt.clf()

        plt.plot(history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train'], loc='upper left')
        plt.savefig(f'{path}/val_loss')
        plt.clf()

        start_new = start_new + timedelta(hours=1)
        end_new = end_new + timedelta(hours=1)
        
    print('done')