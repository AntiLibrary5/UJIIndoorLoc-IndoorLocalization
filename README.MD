# UJIndoorLoc

#### Folders description:
models: contains relevant code for different NN models <br>

saved_models: contains .h5 tensorflow models that have been trained <br>

utils: contains helper scripts <br>

data: contains the dataset <br>

#### NB:
Download the dataset from https://archive.ics.uci.edu/ml/machine-learning-databases/00310/UJIndoorLoc.zip and extract trainingData.csv and validationData.csv to the data/ folder as: <br>
- data/trainingData.csv
- data/validationData.csv

#### Tested with:
Python: 3.7 <br>
DataExploration.py is only for additional info but all details relevant are covered in Report_IndoorLoc.pdf. <br>
Main script: <br>
>> python main.py --Train 0 --Name tf_smallDNN 

#### Follow these steps to set up the env:
Preferably set up a virtual environment at \venv. With conda>4.6, in your Anaconda prompt: <br>
   4.1. >> conda cd ..\[PREFERRED-DIRECTORY] <br>
   4.2. >> conda create -n venv python=3.7 <br>
   4.3. >> conda activate venv <br>
   4.4. >> pip install -r requirements.txt <br>

#### To run: >> python main.py --Train 0 --Name tf_smallDNN 
Depending on the argparse, main.py will either: <br>
  - train, save and evaluate a specific model architecture <br>
  - or, load a saved model and evaluate it <br>

For example: >> python main.py --Train 0 --Name tf_smallDNN <br>
will load the saved model and evaluate it on test set <br>

#### Additional info:
Train, save and evaluate models: >> python main.py --Train 1 --Name tf_smallDNN <br>
    - if args --Train == 1 and --Name = 'name' then train 'name' model on the dataset in ./data where <br>
        'name' represents a specific architecture of model: {manualNN, tf_simpleNN, tf_smallDNN, tf_DNN} <br>
        manualNN: a single node NN created with manual code <br>
        tf_simpleNN: TF based regression and classification models with single node <br>
        tf_smallDNN: TF based regression and classification models with 1 hidden layer and 16 nodes <br>
        tf_DNN: TF based regression and classification models with 2 hidden layer and 64 nodes each <br>
    - After training saves the model in 'saved_models/name.h5' <br>
    - After saving Evaluates the model against LONGITUDE, LATITUDE, BUILDINGID and FLOOR prediction on test set and results are displayed on prompt <br>

Load the saved models and evaluate them: >> python main.py --Train 0 --Name tf_smallDNN <br>

DataExploration.py is only for additional info but all details relevant are covered in Report_IndoorLoc.pdf
