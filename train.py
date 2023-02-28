import tensorflow as tf
import matplotlib.pyplot as plt
from get_dataset import scale_data, pred_length
import os
from sklearn.model_selection import GridSearchCV
from keras.wrappers.scikit_learn import KerasClassifier
import numpy as np

file_name = "./data/dolar_bar_2020-2021.csv"



def build_model():
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Conv1D(256, 3, activation='relu', input_shape=(4, 12), padding='same'))
    model.add(tf.keras.layers.Dense(128, activation='relu'))
    model.add(tf.keras.layers.Conv1D(128, 2, activation='relu', input_shape=(4, 12), padding='same'))
    model.add(tf.keras.layers.Dense(512, activation='relu'))
    model.add(tf.keras.layers.Dropout(0.7))
    # model.add(tf.keras.layers.LSTM(256, return_sequences=True))
    model.add(tf.keras.layers.Dense(256, activation='relu'))
    model.add(tf.keras.layers.LSTM(256))
    model.add(tf.keras.layers.Dense(256, activation='relu'))
    model.add(tf.keras.layers.Dropout(0.7))
    model.add(tf.keras.layers.Dense(128, activation='relu'))
    model.add(tf.keras.layers.Dense(3, activation='softmax'))


    model.compile(loss=tf.keras.losses.CategoricalCrossentropy(),
                  optimizer=tf.keras.optimizers.Adam(),
                  metrics=['accuracy'])
    model.summary()

    return model


def make_folder(date1, date2):
    date_path1 = date1.strftime('%m-%d-%Y %H-%M')
    date_path2 = date2.strftime('%m-%d-%Y %H-%M')
    folder_path = f'./saved_model/From {date_path1} To {date_path2}'
    os.mkdir(folder_path)
    return folder_path


def model_checkpoint(model_path):
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=f'{model_path}/Conv.h5',
        save_weights_only=False,
        monitor='val_accuracy',
        mode='max',
        save_best_only=True,
        verbose=1)
    return model_checkpoint_callback


if __name__ == "__main__":
    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
    path = f'./saved_model/From 3-16-2020 23-35 To 12-31-2020 23-55'
    os.mkdir(path)
    # class_weight = {0: 1.0, 1: 10.0, 2: 10.0}  # Ustawienie wag dla klas


    inputs, outputs = scale_data(file_name)
    print(inputs)
    print(inputs.shape)
    print(outputs)
    print(outputs.shape)
    class_freq = np.sum(outputs, axis=0) / outputs.shape[0]
    class_weight = {i: 1.0 / class_freq[i] for i in range(len(class_freq))}
    print(class_weight)

    model = build_model()

    history = model.fit(inputs, outputs, epochs=1000, batch_size=16,
                        validation_split=0.2,
                        verbose=1,
                        callbacks=[model_checkpoint(path)], class_weight=class_weight)

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

    print('done')
