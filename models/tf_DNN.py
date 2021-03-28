from utils.eval_tf import plotAccLoss

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense, Activation, Input, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import SGD

# -----------------------------------------------------------
# Implements a NN with a 2 hidden layers with 64 nodes each
# for classification and regression task with dropout
# -----------------------------------------------------------

def DNN_reg(xTrain, yTrain):
    callback = EarlyStopping(monitor='loss', patience=2)
    model = Sequential( [Input(shape=xTrain.shape[1]), Dense(units=1), Dense(units=64, activation='relu'), Dropout(0.5), Dense(units=64, activation='relu'), Dropout(0.5), Dense(units=1)  ] )
    model.summary()
    model.compile(optimizer='sgd', loss='mean_squared_error')
    hist = model.fit(xTrain, yTrain, validation_split=0.2, batch_size = 32, epochs = 50, verbose=0, callbacks=[callback])
    plotAccLoss(hist)
    return model

def DNN_clf(xTrain, yTrain):
    callback = EarlyStopping(monitor='loss', patience=2)
    model = Sequential( [Dense(units=1, input_dim=xTrain.shape[1]), Dense(units=64, activation='relu'), Dropout(0.5), Dense(units=64, activation='relu'), Dropout(0.5), Dense(units=5, activation='softmax')   ] )
    model.summary()
    model.compile(optimizer='sgd', loss='sparse_categorical_crossentropy')
    hist = model.fit(xTrain, yTrain,  validation_split=0.2, batch_size = 32, epochs = 50, verbose=0, callbacks=[callback])
    plotAccLoss(hist)
    return model

def run_DNN(X_train, Y_train):
    modelRegDeep_Long = DNN_reg(X_train, Y_train['LONGITUDE'])
    modelRegDeep_Lat = DNN_reg(X_train, Y_train['LATITUDE'])
    modelClfDeep_Build = DNN_clf(X_train, Y_train['BUILDINGID'])
    modelClfDeep_Floor = DNN_clf(X_train, Y_train['FLOOR'])
    return modelRegDeep_Long, modelRegDeep_Lat, modelClfDeep_Build, modelClfDeep_Floor
