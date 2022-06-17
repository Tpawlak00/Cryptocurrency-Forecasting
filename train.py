import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Dropout
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
# from keras.utils import to_categorical
from main import scale_train_data, pred_length


def make_model():
    model = Sequential()
    model.add(Conv1D(filters=16, kernel_size=12, activation='relu', input_shape=(pred_length, 1)))
    model.add(Conv1D(filters=16, kernel_size=12, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(100, activation='softmax'))
    model.add(Dense(3, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model


def train_model(model_name):
    inputs, labels = scale_train_data()
    print(labels.shape)
    inputs = inputs.reshape(len(inputs), len(inputs[0]), 1)
    print(labels.shape)
    model_name.summary()
    model_name.fit(inputs, labels, epochs=1, batch_size=128,  shuffle=True, validation_split=0.2)
    model_name.save('saved_model/Conv_6_18_22')
    print('Model saved')

if __name__ == '__main__':
    crypto_model = make_model()
    train_model(crypto_model)