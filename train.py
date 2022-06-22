from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten, Activation
from keras.layers import Dropout, BatchNormalization
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
import matplotlib.pyplot as plt
from main import scale_data, pred_length
# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"


def make_model():
    model = Sequential()
    model.add(Conv1D(100, 2, activation='relu', input_shape=(pred_length, 1)))
    model.add(Conv1D(50, 2))
    # model.add(Dropout(0.5))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling1D(2))
    model.add(Flatten())
    model.add(Dense(3, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=['accuracy'])

    return model


def train_model(model_name):
    inputs, labels, _, _ = scale_data()
    print(labels.shape)
    model_name.summary()
    history = model_name.fit(inputs, labels, epochs=100, batch_size=32)
    model_name.save('saved_model/Conv_6_22_22')
    print('Model saved')

    plt.plot(history.history['accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

    plt.plot(history.history['loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()


if __name__ == '__main__':
    crypto_model = make_model()
    train_model(crypto_model)
