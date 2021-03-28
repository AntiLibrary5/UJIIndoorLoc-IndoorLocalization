from utils.eval_tf import plotAccLoss

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense, Activation, Input, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import SGD

# -----------------------------------------------------------
# Implements a single node NN for classification and regression task
# -----------------------------------------------------------

def simpleNN_reg(xTrain, yTrain):
    callback = EarlyStopping(monitor='loss', patience=3)
    model = Sequential([ Dense(1, input_dim=xTrain.shape[1]) ])
    model.summary()
    model.compile(optimizer='sgd', loss='mean_squared_error')
    hist = model.fit(xTrain, yTrain, validation_split=0.2, epochs=3, verbose=1, callbacks=[callback])
    plotAccLoss(hist)
    return model

def simpleNN_clf(xTrain, yTrain):
    callback = EarlyStopping(monitor='loss', patience=3)
    model = Sequential( [Dense(units=1, input_dim=xTrain.shape[1]), Dense(units=5, activation='softmax')  ] )
    model.summary()
    model.compile(optimizer='sgd', loss='sparse_categorical_crossentropy')
    hist = model.fit(xTrain, yTrain, validation_split=0.2, verbose=1, epochs=3, callbacks=[callback])
    plotAccLoss(hist)
    return model

def run_simpleNN(X_train, Y_train):
    print("Training...")
    modelReg_Long = simpleNN_reg(X_train, Y_train['LONGITUDE'])
    modelReg_Lat = simpleNN_reg(X_train, Y_train['LATITUDE'])
    modelClf_Build = simpleNN_clf(X_train, Y_train['BUILDINGID'])
    modelClf_Floor = simpleNN_clf(X_train, Y_train['FLOOR'])
    return modelReg_Long, modelReg_Lat, modelClf_Build, modelClf_Floor
