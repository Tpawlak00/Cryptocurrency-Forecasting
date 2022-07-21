<<<<<<< HEAD
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from get_dataset import read_file, one_hot, split_data, pred_length

scaler = MinMaxScaler()
path = './data/BTC_minute.csv'
start_train, end_train = '3/16/2020  23:35:00', '3/01/2022  23:55:00'
start_test = '3/02/2022  00:00:00'
model_path = './saved_model/Conv.h5'


def build_model():
    model_name = tf.keras.models.Sequential()

    model_name.add(tf.keras.layers.Conv1D(32, 2, input_shape=(pred_length, 1)))
    # model.add(tf.keras.layers.AveragePooling1D(2))
    model_name.add(tf.keras.layers.LSTM(16))
    # model.add(tf.keras.layers.Flatten())

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


if __name__ == "__main__":
    data = read_file(path)
    inputs_train, _ = split_data(data['price'], start_train, end_train, start_test)
    outputs_train, _ = one_hot(data['target'], start_train, end_train, start_test)
    model = build_model()

    x_train = inputs_train[0:2000]
    x_train = scaler.fit_transform(x_train)
    y_train = outputs_train[0:2000]
    history = model.fit(x_train, y_train, epochs=1000, batch_size=16, validation_split=0.2, verbose=2)
    model.save(model_path)
    print('Model saved')

    for i in range(2000, len(inputs_train), 2000):
        model = tf.keras.models.load_model(model_path)
        x_train = inputs_train[i:i + 2000]
        x_train = scaler.fit_transform(x_train)
        y_train = outputs_train[i:i + 2000]
        history = model.fit(x_train, y_train, epochs=1000, batch_size=16, validation_split=0.2, verbose=2)
        model.save(model_path)
        print('Model saved')
    print('Finished training')

=======
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Dropout, BatchNormalization
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
import matplotlib.pyplot as plt
from main import scale_data, pred_length
# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"


def make_model():
    model = Sequential()
    model.add(Conv1D(120, 3, activation='relu', input_shape=(pred_length, 1)))
    model.add(BatchNormalization())
    model.add(Conv1D(30, 3, activation='relu'))
    model.add(BatchNormalization())
    # model.add(MaxPooling1D(2))
    model.add(Flatten())
    # model.add(Dense(64,activation='softmax'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(3, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=['accuracy'])

    return model


def train_model(model_name):
    inputs, labels, _, _ = scale_data()
    print(labels.shape)
    model_name.summary()
    history = model_name.fit(inputs, labels, epochs=1, batch_size=32, shuffle='True', validation_split=0.2)
    model_name.save('saved_model/Conv_6_27_22')
    print('Model saved')

>>>>>>> 51264b1cddbca5283a2d8c1d4221b167941e53aa
    plt.plot(history.history['accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train'], loc='upper left')
    plt.show()

<<<<<<< HEAD
=======
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['validation'], loc='upper left')
    plt.show()

>>>>>>> 51264b1cddbca5283a2d8c1d4221b167941e53aa
    plt.plot(history.history['loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train'], loc='upper left')
    plt.show()
<<<<<<< HEAD
=======

    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['validation'], loc='upper left')
    plt.show()


if __name__ == '__main__':
    crypto_model = make_model()
    train_model(crypto_model)
>>>>>>> 51264b1cddbca5283a2d8c1d4221b167941e53aa
