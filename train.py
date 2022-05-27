from tensorflow.keras import Sequential, regularizers
from tensorflow.keras.layers import LSTM, Dense, Dropout, Activation
from main import scale_train_data


def make_model():
    model = Sequential()

    model.add(LSTM(units=32, return_sequences=True, input_shape=(None, 1), kernel_regularizer=regularizers.l2(0.0001)))
    model.add(Dropout(0.5))

    model.add(LSTM(units=64, return_sequences=True, kernel_regularizer=regularizers.l2(0.0001)))
    model.add(Dropout(0.6))

    model.add(LSTM(units=128, return_sequences=True, kernel_regularizer=regularizers.l2(0.0001)))
    model.add(Dropout(0.6))

    model.add(LSTM(units=64, return_sequences=True, kernel_regularizer=regularizers.l2(0.0001)))
    model.add(Dropout(0.6))

    model.add(LSTM(units=32, return_sequences=True, kernel_regularizer=regularizers.l2(0.0001)))
    model.add(Dropout(0.6))

    model.add(Dense(units=1))

    return model


def train_model(model_name):
    inputs, labels = scale_train_data()
    model_name.compile(loss='mse', optimizer='adam')
    model_name.summary()
    model_name.fit(inputs, labels, epochs=1, batch_size=32)
    model_name.save('saved_model/Model_11')
    print('Model saved')


crypto_model = make_model()
train_model(crypto_model)