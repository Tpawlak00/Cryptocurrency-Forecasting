import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from get_dataset import scale_data
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import xgboost as xgb
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import GridSearchCV
import pickle

pred_length = 12
scaler = MinMaxScaler()
file_name = "./data/BTC_2022.csv"


def create_files():
    path = "./saved_model"
    inputs, _ = scale_data(file_name)
    for dirpath, dirnames, filenames in os.walk(path):
        try:
            dirname = dirpath.split(os.path.sep)[1]
        except IndexError:
            continue
        print(dirname)
        model = tf.keras.models.load_model(f'./saved_model/{dirname}/Conv.h5')
        pred = model.predict(np.array(inputs))
        data = np.array(pred)
        np.savetxt(f'./XBoost_data/{dirname}_predictions.csv', data, delimiter=',')


def concat_predictions():
    path = "./XBoost_data"
    filename = os.listdir(path)
    df1 = pd.read_csv(f'./XBoost_data/{filename[0]}')

    for i in range(1, len(filename)):
        print(filename[i])
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
        if max_index == 0:
            pred_array[i] = [1, 0, 0]
        elif max_index == 1:
            pred_array[i] = [0, 1, 0]
        elif max_index == 2:
            pred_array[i] = [0, 0, 1]

    return pred_array


if __name__ == "__main__":
    # create_files() # Uncomment to create predictions of each model
    # concat_predictions() # Uncomment to create one big dataset of all models predictions
    #
    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
    df = pd.read_csv('./XBoost_data/Superset.csv', sep=',', header=None, low_memory=False)
    inputs = df.to_numpy()
    _, outputs = scale_data(file_name)

    X_train, X_calib, y_train, y_calib = train_test_split(inputs, outputs)

    #####################
    # XGBOOST
    # ####################
    clf = xgb.XGBClassifier(max_depth=6, min_child_weight=14, learning_rate = 0.1, gamma=0.76, n_estimators=60,
                            subsample=0.53, colsample_bytree=0.33, nthread=4, scale_pos_weight=1, seed=60)
    parameters = {
        'n_estimators': [10,20,30,40,50,60,70,80,90,100,120,140,160,180,200]

    }
    # grid = GridSearchCV(estimator=clf, param_grid=parameters, cv=10, scoring='f1_macro', n_jobs=-1, verbose=10)
    # grid.fit(X_train, y_train)
    # print(grid.best_params_, grid.best_score_)

    clf.fit(X_train, y_train)
    # y_pred = clf.predict(X_test)
    # accuracy = accuracy_score(y_test, y_pred)
    # print(accuracy * 100.0)
    # print("Accuracy: %.2f%%" % (accuracy * 100.0))
    clf.save_model("xgb_12model5.txt")

    #####################
    # ADABOOST
    # ####################

    dt = DecisionTreeClassifier(max_depth = 23, max_features=22,min_samples_split=0.1,
                                random_state=1)
    params_dt = {
        'max_depth': [19,20,21,22,23,24,25],
        'max_features': [19,20,21,22,23,24,25],
        'min_samples_split': [0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.09,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8]
    }
    # print(dt.get_params())
    # grid_dt = GridSearchCV(estimator=dt, param_grid=params_dt, scoring='f1_macro', cv=10, n_jobs=-1, verbose=10)
    # grid_dt.fit(X_train, y_train)
    # print(grid_dt.best_estimator_)

    y_train_vector = []
    for i in range(len(y_train)):
        max_val = np.max(y_train[i])
        max_index = np.where(y_train[i] == max_val)[0]
        if max_index == 0:
            y_train_vector.append(0)
        elif max_index == 1:
            y_train_vector.append(1)
        elif max_index == 2:
            y_train_vector.append(2)
    #
    ada = AdaBoostClassifier(base_estimator=dt, n_estimators=420, random_state=1)
    #
    # grid_ada = GridSearchCV(estimator=ada, param_grid={'n_estimators':[390,400,410,420,430,440,450]}, scoring='f1_macro', cv=10, n_jobs=-1, verbose=10)
    # grid_ada.fit(X_train, y_train_vector)
    # print(grid_ada.best_estimator_)
    # #
    #
    # ada.fit(X_train, y_train_vector)
    # calibrated_clf = CalibratedClassifierCV(base_estimator=ada, method='sigmoid', cv='prefit')
    # calibrated_clf.fit(X_calib, y_calib)
    # pickle.dump(ada, open("ada_12model15.sav", 'wb'))
    # pred =ada.predict_proba(X_calib)
    # print(pred)