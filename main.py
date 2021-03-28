# import custom scripts
import utils.preprocess as preprocess
import utils.eval_tf as eval_tf
import models.tf_simpleNN as tf_simpleNN
import models.tf_smallDNN as tf_smallDNN
import models.tf_DNN as tf_DNN
import models.manualNN as manualNN

# import standard libraries
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from tensorflow import keras

import argparse

# -----------------------------------------------------------
# Depending on the argparse, either:
#   - train, save and evaluate a specific model architecture
#   - or, load a saved model and evaluate it
#
# by Vaibhav Arora
# email vaibhav.arora@u-psud.fr
# -----------------------------------------------------------

"""
Loading the UJIndoorLoc dataset:
    - trainingData.csv is used as the train/val set
    - validationData.csv is used as the test data only at the end to evaluate a model
"""
print("Loading data...")
trainData = pd.read_csv("data/trainingData.csv") #train/val set
testData = pd.read_csv("data/validationData.csv") #test set

"""
Data pre-processing:
    - Input features are scaled
    - Target train features LONGITUDE and LATITUDE are standardized
"""
print("Pre-processing data...")
#Loading and scaling input train features
X_train = trainData.iloc[:,:520]
X_train = preprocess.invertRSS(X_train)
X_train = preprocess.scaleRSS(X_train)

#Loading and scaling input test features
X_test = testData.iloc[:,:520]
X_test = preprocess.invertRSS(X_test)
X_test = preprocess.scaleRSS(X_test)

#Loading target train features
Y_train = trainData.iloc[:,520:]

#Scaler object for longitude later used to invert the predictions
#back to original coordinates to evaluate against test data
scalerLong = StandardScaler()
Y_train_long = Y_train['LONGITUDE'].values.reshape(len(Y_train), 1) #scikit requirement for arrays
scalerLong.fit(Y_train_long)
Y_train['LONGITUDE'] = scalerLong.transform(Y_train_long)

#scaler object for latitude later used to invert the predictions
#back to original coordinates to evaluate against test data
scalerLat = StandardScaler()
Y_train_lat = Y_train['LATITUDE'].values.reshape(len(Y_train), 1) #scikit requirement for arrays
scalerLat.fit(Y_train_lat)
Y_train['LATITUDE'] = scalerLat.transform(Y_train_lat)

#Loading target test features
Y_test = testData.iloc[:,520:]

def main():
    argparser = argparse.ArgumentParser(
        description=__doc__)
    argparser.add_argument(
        '-t', '--Train',
        metavar='S',
        default=0,
        type=int,
        help='Train model specified by --name arg, int-type: {0, 1} (default: 0)')
    argparser.add_argument(
        '-n', '--Name',
        metavar='N',
        default=0,
        type=str,
        help='Name of model to be trained, str-type: {manualNN, tf_simpleNN, tf_smallDNN, tf_DNN}, (default: 0)')
    args = argparser.parse_args()

    TRAIN_MODEL = args.Name
    TRAIN_FLAG = args.Train

    """
    Train, save and evaluate models:
        - if args --Train == 1 and --Name = 'name' then train 'name' model on the dataset where
            'name' represents a specific architecture of model: {manualNN, tf_simpleNN, tf_smallDNN, tf_DNN}
            manualNN: a single node NN created with manual code
            tf_simpleNN: TF based regression and classification models with single node
            tf_smallDNN: TF based regression and classification models with 1 hidden layer and 16 nodes
            tf_DNN: TF based regression and classification models with 2 hidden layer and 64 nodes each
        - After training saves the model in 'saved_models/name.h5'
        - After saving Evaluates the model against LONGITUDE, LATITUDE, BUILDINGID and FLOOR prediction on test set
    """
    if TRAIN_FLAG == 1:
        if TRAIN_MODEL == 'manualNN':
            print("Training manualNN...")
            # Manually coded NN with a single node
            # MSE for REGRESSION
            # NLL for CLASSIFICATION
            n_epochs = 10
            step_size = 0.001
            W_Build, b_Build, W_Floor, b_Floor, W_Long, b_Long, W_Lat, b_Lat = manualNN.Run_manualNN(n_epochs, step_size, X_train, Y_train, X_test, Y_test)
            print("Predicted: ", np.argmax(W_Build@X_test.iloc[1] + b_Build))
            print("True label: ", Y_test['BUILDINGID'].iloc[1])

        elif TRAIN_MODEL == 'tf_simpleNN':
            print("Training tf_simpleNNs...")
            # TF based regression and classification models with single node
            modelReg_Long, modelReg_Lat, modelClf_Build, modelClf_Floor = tf_simpleNN.run_simpleNN(X_train, Y_train)
            print("Saving models...")
            # Save models
            modelReg_Long.save('saved_models/modelReg_Long.h5')
            modelReg_Lat.save('saved_models/modelReg_Lat.h5')
            modelClf_Build.save('saved_models/modelClf_Build.h5')
            modelClf_Floor.save('saved_models/modelClf_Floor.h5')
            print("Evaluating...")
            # Evaluate
            eval_tf.evaluate_std(modelReg_Long, modelReg_Lat, modelClf_Build, modelClf_Floor, X_test, Y_test, scalerLong, scalerLat)

        elif TRAIN_MODEL == 'tf_smallDNN':
            print("Training tf_smallDNNs...")
            # TF based regression and classification models with 1 hidden layer and 16 nodes
            modelRegMed_Long, modelRegMed_Lat, modelClfMed_Build, modelClfMed_Floor = tf_smallDNN.run_smallDNN(X_train, Y_train)
            print("Saving models...")
            # Save models
            modelRegMed_Long.save('saved_models/modelRegMed_Long.h5')
            modelRegMed_Lat.save('saved_models/modelRegMed_Lat.h5')
            modelClfMed_Build.save('saved_models/modelClfMed_Build.h5')
            modelClfMed_Floor.save('saved_models/modelClfMed_Floor.h5')
            print("Evaluating...")
            # Evaluate models
            eval_tf.evaluate_std(modelRegMed_Long, modelRegMed_Lat, modelClfMed_Build, modelClfMed_Floor, X_test, Y_test, scalerLong, scalerLat)

        elif TRAIN_MODEL == 'tf_DNN':
            print("Training tf_DNNs...")
            # TF based regression and classification models with 2 hidden layer and 64 nodes each
            modelRegDeep_Long, modelRegDeep_Lat, modelClfDeep_Build, modelClfDeep_Floor = tf_DNN.run_DNN(X_train, Y_train)
            print("Saving models...")
            # Save models
            modelRegDeep_Long.save('saved_models/modelRegDeep_Long.h5')
            modelRegDeep_Lat.save('saved_models/modelRegDeep_Lat.h5')
            modelClfDeep_Build.save('saved_models/modelClfDeep_Build.h5')
            modelClfDeep_Floor.save('saved_models/modelClfDeep_Floor.h5')
            print("Evaluating...")
            # Evaluate models
            eval_tf.evaluate_std(modelRegDeep_Long, modelRegDeep_Lat, modelClfDeep_Build, modelClfDeep_Floor, X_test, Y_test, scalerLong, scalerLat)
        else:
            print("Invalid model name specified. Please select one from {manualNN, tf_simpleNN, tf_smallDNN, tf_DNN}")


    """
    Load the saved models and evaluate them
    """
    if TRAIN_FLAG == 0:
        if TRAIN_MODEL == 'manualNN':
            print("Training manualNN...")
            # Manually coded NN with a single node
            # MSE for REGRESSION
            # NLL for CLASSIFICATION
            W_Build, b_Build, W_Floor, b_Floor, W_Long, b_Long, W_Lat, b_Lat = manualNN.Run_manualNN(n_epochs, step_size)

        elif TRAIN_MODEL == 'tf_simpleNN':
            print("Loading  tf_simpleNNs models...")
            # Save models
            modelReg_Long = keras.models.load_model('saved_models/modelReg_Long.h5')
            modelReg_Lat = keras.models.load_model('saved_models/modelReg_Lat.h5')
            modelClf_Build = keras.models.load_model('saved_models/modelClf_Build.h5')
            modelClf_Floor = keras.models.load_model('saved_models/modelClf_Floor.h5')
            print("Evaluating...")
            # Evaluate
            eval_tf.evaluate_std(modelReg_Long, modelReg_Lat, modelClf_Build, modelClf_Floor, X_test, Y_test, scalerLong, scalerLat)

        elif TRAIN_MODEL == 'tf_smallDNN':
            print("Loading  tf_smallDNNs models...")
            # Save models
            modelRegMed_Long = keras.models.load_model('saved_models/modelRegMed_Long.h5')
            modelRegMed_Lat = keras.models.load_model('saved_models/modelRegMed_Lat.h5')
            modelClfMed_Build = keras.models.load_model('saved_models/modelClfMed_Build.h5')
            modelClfMed_Floor = keras.models.load_model('saved_models/modelClfMed_Floor.h5')
            print("Evaluating...")
            # Evaluate models
            eval_tf.evaluate_std(modelRegMed_Long, modelRegMed_Lat, modelClfMed_Build, modelClfMed_Floor, X_test, Y_test, scalerLong, scalerLat)

        elif TRAIN_MODEL == 'tf_DNN':
            print("Loading  tf_DNNs models...")
            # Save models
            modelRegDeep_Long = keras.models.load_model('saved_models/modelRegDeep_Long.h5')
            modelRegDeep_Lat = keras.models.load_model('saved_models/modelRegDeep_Lat.h5')
            modelClfDeep_Build = keras.models.load_model('saved_models/modelClfDeep_Build.h5')
            modelClfDeep_Floor = keras.models.load_model('saved_models/modelClfDeep_Floor.h5')
            print("Evaluating...")
            # Evaluate models
            eval_tf.evaluate_std(modelRegDeep_Long, modelRegDeep_Lat, modelClfDeep_Build, modelClfDeep_Floor, X_test, Y_test, scalerLong, scalerLat)
        else:
            print("Invalid model name specified. Please select one from {manualNN, tf_simpleNN, tf_smallDNN, tf_DNN}")


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print('User interrupt.')
