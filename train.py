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

    plt.plot(history.history['accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train'], loc='upper left')
    plt.show()

    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['validation'], loc='upper left')
    plt.show()

    plt.plot(history.history['loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train'], loc='upper left')
    plt.show()

    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['validation'], loc='upper left')
    plt.show()


if __name__ == '__main__':
    crypto_model = make_model()
    train_model(crypto_model)
