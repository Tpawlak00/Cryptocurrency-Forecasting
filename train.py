import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from get_dataset import scale_data, pred_length
from datetime import datetime, timedelta
import os

file_name = "./data/BTC_2020.csv"

def build_model():
    model_name = tf.keras.models.Sequential()

    model_name.add(tf.keras.layers.Conv1D(256, 3, input_shape=(pred_length, 1), activation='relu'))
    model_name.add(tf.keras.layers.Dropout(0.3))
    # model_name.add(tf.keras.layers.Conv1D(128, 2, activation='relu'))
    # model_name.add(tf.keras.layers.Dropout(0.3))
    model_name.add(tf.keras.layers.Dense(128, activation='relu'))
    model_name.add(tf.keras.layers.Dropout(0.3))
    # model_name.add(tf.keras.layers.Dropout(0.2))
    # model_name.add(tf.keras.layers.LSTM(128, return_sequences=True))
    # model_name.add(tf.keras.layers.Dropout(0.2))
    model_name.add(tf.keras.layers.LSTM(256, return_sequences=True))
    model_name.add(tf.keras.layers.Dropout(0.3))
    model_name.add(tf.keras.layers.LSTM(128))
    # model_name.add(tf.keras.layers.Dropout(0.2))
    # model_name.add(tf.keras.layers.Flatten())

    model_name.add(tf.keras.layers.Dense(128, activation='relu'))

    # model_name.add(tf.keras.layers.Dropout(0.2))

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

    inputs, outputs = scale_data(file_name)
    print(inputs)
    print(outputs)

    model = build_model()
    # model = tf.keras.models.load_model(f'{path}/Conv.h5')
    history = model.fit(inputs, outputs, epochs=350, batch_size=512,
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

    print('done')