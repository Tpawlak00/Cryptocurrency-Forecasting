import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
# import tensorflow as tf
from get_dataset import scale_data
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import xgboost as xgb
from sklearn.model_selection import GridSearchCV

pred_length = 12
scaler = MinMaxScaler()
file_name = "./data/BTC_2022.csv"


# def create_training_set():
#     path = "./saved_model"
#     inputs, _ = scale_data(file_name)
#
#     predictions = []
#     to_file = []
#     for i in range(0, len(inputs)):
#         print(f'Creating data: {i}/{len(inputs)}')
#         for dirpath, dirnames, filenames in os.walk(path):
#             try:
#                 dirname = dirpath.split(os.path.sep)[1]
#             except IndexError:
#                 continue
#             model = tf.keras.models.load_model(f'./saved_model/{dirname}/Conv.h5')
#             pred = model.predict(np.array([inputs[i]]))
#             predictions.extend(pred[0])
#         to_file.append(predictions)
#         predictions = []
#     data = np.array(to_file)
#     np.savetxt('./data/XBoost_training_data.csv', data, delimiter=',')


def create_files():
    path = "./saved_model"
    inputs, _ = scale_data(file_name)
    for dirpath, dirnames, filenames in os.walk(path):
        try:
            dirname = dirpath.split(os.path.sep)[1]
        except IndexError:
            continue
        model = tf.keras.models.load_model(f'./saved_model/{dirname}/Conv.h5')
        pred = model.predict(np.array(inputs))
        data = np.array(pred)
        np.savetxt(f'./XBoost_data/{dirname}_predictions.csv', data, delimiter=',')


def concat_predictions():
    path = "./XBoost_data"
    filename = os.listdir(path)
    df1 = pd.read_csv(f'./XBoost_data/{filename[0]}')

    for i in range(1, len(filename)):
        print(f'Files done: {i}/{len(filename)}')
        df2 = pd.read_csv(f'./XBoost_data/{filename[i]}')
        df1 = pd.concat([df1, df2], axis=1)

    df1.to_csv('./XBoost_data/Superset.csv', index=False)


def filter(pred_array):
    for i in range(len(pred_array)):
        # get index of max value
        max_val = np.max(pred_array[i])
        # Find index position of maximum value using where()
        max_index = np.where(pred_array[i] == max_val)[0]
        sort_values = np.sort(pred_array[i])[::-1]

        # if max value is greater by 0.2 than the rest of values set categories
        if max_val - sort_values[1] > 0.9 and max_val - sort_values[2] > 0.9:
            if max_index == 0:
                pred_array[i] = [1, 0, 0]
            elif max_index == 1:
                pred_array[i] = [0, 1, 0]
            elif max_index == 2:
                pred_array[i] = [0, 0, 1]
        # else choose "hold" option
        else:
            pred_array[i] = [0, 1, 0]

    return pred_array


if __name__ == "__main__":
    # create_files() # Uncomment to create predictions of each model
    # concat_predictions() # Uncomment to create one big dataset of all models predictions
    #
    df = pd.read_csv('./XBoost_data/Superset.csv', sep=',', header=None, low_memory=False)
    inputs = df.to_numpy()
    _, outputs = scale_data(file_name)

    seed = 7
    test_size = 0.33
    X_train, X_test, y_train, y_test = train_test_split(inputs, outputs, test_size=test_size, random_state=seed)
    print(X_train)
    print(type(X_train))
    print(type(y_train))

    parameters = {
        'seed': [20,40,60,80,100,120,130]
    }
    print("staring search")
    clf = xgb.XGBClassifier(max_depth=3, min_child_weight=5, learning_rate = 0.1, gamma=0.4, n_estimators=70,
                            subsample=0.75, colsample_bytree=0.7, nthread=4, scale_pos_weight=1, seed=60)

    # grid = GridSearchCV(clf, param_grid=parameters, scoring='f1_macro', n_jobs=4, verbose=10)
    clf.fit(X_train, y_train)
    # grid.fit(X_train, y_train)
    # print(grid.best_params_, grid.best_score_)

    y_pred = clf.predict(X_test)
    predictions = filter(y_pred)
    print(predictions)
    accuracy = accuracy_score(y_test, predictions)
    print(accuracy * 100.0)
    print("Accuracy: %.2f%%" % (accuracy * 100.0))
    clf.save_model("model.txt")
