from utils.eval_tf import plotAccLoss

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense, Activation, Input, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import SGD

# -----------------------------------------------------------
# Implements a NN with a single hidden layer with 16 nodes
# for classification and regression task
# -----------------------------------------------------------

def smallDNN_reg(xTrain, yTrain):
    callback = EarlyStopping(monitor='loss', patience=2)
    model = Sequential( [Input(shape=xTrain.shape[1]), Dense(units=1), Dense(units=16, activation='relu'), Dense(units=1)  ] )
    model.summary()
    model.compile(optimizer='sgd', loss='mean_squared_error')
    hist = model.fit(xTrain, yTrain, validation_split=0.2, batch_size=32, epochs = 50, verbose=1, callbacks=[callback])
    plotAccLoss(hist)
    return model

def smallDNN_clf(xTrain, yTrain):
    callback = EarlyStopping(monitor='loss', patience=2)
    model = Sequential( [Dense(units=1, input_dim=xTrain.shape[1]), Dense(units=16, activation='relu'), Dropout(0.5), Dense(units=5, activation='softmax')   ] )
    model.summary()
    model.compile(optimizer='sgd', loss='sparse_categorical_crossentropy')
    hist = model.fit(xTrain, yTrain,  validation_split=0.2, batch_size = 32, epochs = 50, verbose=0, callbacks=[callback])
    plotAccLoss(hist)
    return model

def run_smallDNN(X_train, Y_train):
    modelRegMed_Long = smallDNN_reg(X_train, Y_train['LONGITUDE'])
    modelRegMed_Lat = smallDNN_reg(X_train, Y_train['LATITUDE'])
    modelClfMed_Build = smallDNN_clf(X_train, Y_train['BUILDINGID'])
    modelClfMed_Floor = smallDNN_clf(X_train, Y_train['FLOOR'])
    return modelRegMed_Long, modelRegMed_Lat, modelClfMed_Build, modelClfMed_Floor
