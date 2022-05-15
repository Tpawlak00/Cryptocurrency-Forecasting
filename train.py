from tensorflow.keras import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Activation
from main import scale_train_data


def make_model():
    model = Sequential()

    model.add(LSTM(units=32, return_sequences=True, input_shape=(None, 1)))
    model.add(Dropout(0.2))

    model.add(LSTM(units=64, return_sequences=True))
    model.add(Dropout(0.2))

    model.add(LSTM(units=128, return_sequences=True))
    model.add(Dropout(0.2))

    model.add(LSTM(units=64, return_sequences=True))
    model.add(Dropout(0.2))

    model.add(LSTM(units=32, return_sequences=True))
    model.add(Dropout(0.2))

    model.add(Dense(units=1))

    return model


def train_model(model_name):
    inputs, labels = scale_train_data()
    model_name.compile(loss='mse', optimizer='adam')
    model_name.summary()
    model_name.fit(inputs, labels, epochs=1, batch_size=1)
    model_name.save('saved_model/Model_5')
    print('Model saved')


crypto_model = make_model()
train_model(crypto_model)
