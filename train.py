import tensorflow as tf
import matplotlib.pyplot as plt
from get_dataset import scale_data, pred_length
import os

file_name = "./data/BTC_WMA_2020.csv"



def build_model():
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Conv1D(12, 3, activation='relu', input_shape=(4, 12), padding='same'))
    model.add(tf.keras.layers.Dense(64, activation='relu'))
    # model.add(tf.keras.layers.Dropout(0.2))
    # model.add(tf.keras.layers.Conv1D(filters=128, kernel_size=2, activation='relu', padding='same'))
    model.add(tf.keras.layers.Dense(128, activation='relu'))
    model.add(tf.keras.layers.Dropout(0.4))
    # model.add(tf.keras.layers.LSTM(128, return_sequences=True))
    # model.add(tf.keras.layers.Dense(512, activation='relu'))
    # model.add(tf.keras.layers.Dropout(0.2))
    model.add(tf.keras.layers.LSTM(256))
    model.add(tf.keras.layers.Dense(128, activation='relu'))
    model.add(tf.keras.layers.Dropout(0.3))
    model.add(tf.keras.layers.Dense(64, activation='relu'))
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

    inputs, outputs = scale_data(file_name)
    print(inputs)
    print(inputs.shape)
    print(outputs)
    print(outputs.shape)

    model = build_model()
    # model = tf.keras.models.load_model(f'{path}/Conv.h5')
    history = model.fit(inputs, outputs, epochs=700, batch_size=64,
                        validation_split=0.2,
                        verbose=1,
                        callbacks=[model_checkpoint(path)])
    # print(history.history)

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
