import pandas as pd
import time
import numpy as np
import matplotlib.pyplot as plt
from threading import Thread
from itertools import product
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras import backend as K


def plot(y_true, y_pred, title=None, path=None, saved=False, showed=False):
    fig = plt.figure()
    plt.plot(y_true, 'b-', label='True values')
    plt.plot(y_pred, 'r-', label='Predicted values')
    plt.legend(loc='upper left')
    plt.title(title)
    if saved:
        plt.savefig(path)
    if showed:
        plt.show()
    plt.close()


def time_series_to_supervised(data, lag=5):
    data = pd.DataFrame(data)
    columns = [data.shift(i) for i in range(1, lag+1)]    
    columns.append(data)
    data = pd.concat(columns, axis=1)
    data.fillna(0, inplace=True)
    return data.values


def load_and_transform(mins, dev, lag):
    df = pd.read_csv("../data/google_cloud_trace/{}Minutes_6176858948.csv".format(mins))
    if dev == 'cpu':
        data = df.iloc[:, 3]
    elif dev == 'ram':
        data = df.iloc[:, 4]
    
    train_indices = int(0.8 * len(data))
    train_ts, test_ts = np.array(data[:train_indices]), np.array(data[train_indices:])
    
    mean = np.mean(train_ts)
    std = np.std(train_ts)

    scaled_train_ts = (train_ts - mean) / std
    scaled_test_ts = (test_ts - mean) / std

    return time_series_to_supervised(scaled_train_ts, lag), time_series_to_supervised(scaled_test_ts, lag), mean, std


def fit_lstm(train_data, batch_size, nb_epoch, neurons):
    X, y = train_data[:, 0:-1], train_data[:, -1]
    X = X.reshape(X.shape[0], 1, X.shape[1])
    model = Sequential()
    model.add(LSTM(neurons, batch_input_shape=(batch_size, X.shape[1], X.shape[2]), stateful=True))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    for _ in range(nb_epoch):
        model.fit(X, y, epochs=1, batch_size=batch_size, verbose=0, shuffle=False)
        model.reset_states()
    return model


def forecast_lstm(model, batch_size, X):
    X = X.reshape(X.shape[0], 1, X.shape[1])
    yhat = model.predict(X, batch_size=batch_size)
    return yhat


def experiment(*args):
    dev = args[0]
    mins = args[1]
    time_step = args[2]
    neurons = args[3]
    n_epochs = args[4]
    #num_epochs = [500, 1000, 3000]
    train_scaled, test_scaled, mean, std = load_and_transform(mins, dev, time_step)
    y_true = test_scaled[:, -1]
    y_pred = []

    start = time.time()
    for _ in range(10):
        lstm_model = fit_lstm(train_scaled, 1, n_epochs, neurons)
        y_pred.append(forecast_lstm(lstm_model, 1, test_scaled[:, 0:-1]))
        K.clear_session()
    end = time.time()

    y_pred = np.array(y_pred)    
    y_pred = np.mean(y_pred, axis=0)    

    y_true = y_true * std + mean
    y_pred = y_pred * std + mean

    title = "Prediction {} {}".format(dev, mins)
    error = mean_absolute_error(y_true, y_pred)
    plot(y_true, 
        y_pred, 
        title=title + "\nMAE: {}".format(error), 
        path="../result/lstm/ct/image/{}_time_{}_neurons_{}_epochs_{}.png".format(title, str((end - start) / 10.0), neurons, n_epochs).replace(" ", "_"), 
        saved=True, 
        showed=False)
    output = np.hstack((y_true.reshape((-1, 1)), y_pred.reshape((-1, 1))))
    df = pd.DataFrame(output)
    df.to_csv("../result/lstm/ct/csv/{}_time_{}_neurons_{}_epochs_{}.png".format(title, str(end - start), neurons, n_epochs).replace(" ", "_"))


if __name__ == '__main__':
    # device = ['cpu', 'ram']
    # minute = [5, 10]
    # time_step = [5, 10]
    # n_model = [120, 150]

    device = ['cpu', 'ram']
    minute = [5, 10]
    time_step = [5, 10]
    neurons = [10, 100]
    n_epochs = [3000]

    threads = []
    model_name = 'boosting svr'
    for dev, mins, t, nn, ne in product(device, minute, time_step, neurons, n_epochs):
        experiment(dev, mins, t, nn, ne)


