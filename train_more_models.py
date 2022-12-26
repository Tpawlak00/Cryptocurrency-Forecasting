import os
import tensorflow as tf
from get_dataset import scale_data
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

file_name = "./data/BTC_2021.csv"


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
    inputs, outputs = scale_data(file_name)
    start_new = datetime(2021, 1, 1, 0, 0)
    end_new = datetime(2021, 1, 31, 0, 0)

    for i in range(0, len(inputs), 8640):
        x_train, y_train = inputs[i:8640+i], outputs[i:8640+i]
        model = tf.keras.models.load_model(f'{path}/Conv.h5')
        path = make_folder(start_new, end_new)
        history = model.fit(x_train, y_train, epochs=150, batch_size=32,
                            validation_split=0.2,
                            verbose=1, callbacks=[model_checkpoint(path)])

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

        start_new = start_new + timedelta(hours=720)
        end_new = end_new + timedelta(hours=720)
